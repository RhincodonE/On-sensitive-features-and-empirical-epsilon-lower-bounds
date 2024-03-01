import argparse
import os
import numpy as np
import torch
from models import resnet18  # Ensure this is your PyTorch implementation
from utils import make_data, make_canary_data, save_observations, compute_statistics, compute_empirical_epsilon,train_model, audit_model

NUM_CLASSES = 7
NUM_MODELS = 100
MODEL_SAVE_PATH = './models'
NUM_OBSERVATIONS = 10
EPOCHS = 100
BATCH = 20
LR = 0.0002
DELTA = 1e-5  # Privacy parameter

def main():
    parser = argparse.ArgumentParser(description="Train and audit models with differential privacy")
    parser.add_argument("--noise", type=float, default=0.6, help="Noise multiplier for DP")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Desired epsilon")
    parser.add_argument("--data_type", type=str, default='processed', help="Type of data to use")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset1 = make_data(data_type=args.data_type)
    dataset2 = make_data(data_type=args.data_type, remove_id_class=6)
    canary_images, canary_labels = make_canary_data(data_type='original', id_class=6)

    all_O = []
    all_O_prime = []
    all_epsilon_1 = []
    all_epsilon_2 = []

    for i in range(NUM_MODELS):
        print(f"Training model {i+1}/{NUM_MODELS}...")

        # Train model without canary
        model_1, optimizer_1, criterion_1, privacy_engine_1 = resnet18(num_classes=NUM_CLASSES,batch_size=BATCH,epochs=EPOCHS,input_shape=(1, 256, 256),data_loader=dataset1[0],desired_epsilon = args.epsilon)
        model_without_canary = train_model(model_1, dataset1[0], dataset1[1], criterion_1, optimizer_1, EPOCHS)
        epsilon_1 = privacy_engine_1.get_epsilon(delta=DELTA)
        all_epsilon_1.append(epsilon_1)
        print(f"Privacy Budget for model_1: ε = {epsilon_1}")

        # Train model with canary
        model_2, optimizer_2, criterion_2, privacy_engine_2 = resnet18(num_classes=NUM_CLASSES,batch_size=BATCH,epochs=EPOCHS, input_shape=(1, 256, 256),data_loader=dataset2[0],desired_epsilon = args.epsilon)
        model_with_canary = train_model(model_2, dataset2[0], dataset2[1], criterion_2, optimizer_2, EPOCHS)
        epsilon_2 = privacy_engine_2.get_epsilon(delta=DELTA)
        all_epsilon_2.append(epsilon_2)
        print(f"Privacy Budget for model_2: ε = {epsilon_2}")


        torch.save(model_without_canary.state_dict(), os.path.join(MODEL_SAVE_PATH, f'model_without_canary_{i+1}.pth'))
        torch.save(model_with_canary.state_dict(), os.path.join(MODEL_SAVE_PATH, f'model_with_canary_{i+1}.pth'))

        # Collect and save observations
        for j in range(len(canary_images)):
            canary = (canary_images[j], canary_labels[j])
            O, O_prime = [], []
            for t in range(NUM_OBSERVATIONS):
                loss_without_canary, loss_with_canary = audit_model(canary, model_without_canary, model_with_canary, criterion_1)
                O.append(loss_without_canary)
                O_prime.append(loss_with_canary)

            all_O.extend(O)
            all_O_prime.extend(O_prime)
            save_observations(O, O_prime, i+1, j+1)

    threshold, FPR, FNR = compute_statistics(all_O, all_O_prime)
    empirical_epsilon = compute_empirical_epsilon(FPR, FNR)

    average_epsilon_1 = np.mean(all_epsilon_1)
    average_epsilon_2 = np.mean(all_epsilon_2)

    results = (f"Decision Threshold: {threshold}\n"
               f"False Positive Rate: {FPR}\n"
               f"False Negative Rate: {FNR}\n"
               f"Empirical Epsilon: {empirical_epsilon}\n"
               f" Privacy Budget for models without canary: ε = {all_epsilon_1[-1]}\n"
               f" Privacy Budget for models with canary: ε = {all_epsilon_2[-1]}\n"
               f" set privacy budget: ε = {args.epsilon}\n")

    print(results)
    with open(f"{args.data_type}.txt", 'a') as file:
        file.write(results + "\n")

if __name__ == "__main__":
    main()
