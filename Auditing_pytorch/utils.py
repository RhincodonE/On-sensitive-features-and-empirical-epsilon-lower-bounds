import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import os

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def make_data(data_type='original', remove_id_class=None, test_size=0.2, num_augmented_images=5000):
    if data_type == 'original':
        original_images = np.load('./Data/original_images.npy').astype(np.float32) / 255.0
        original_labels = np.load('./Data/JAFFE_labels.npy').astype(int)
        id_labels = np.load('./Data/ID_labels.npy').astype(int)
    elif data_type == 'processed':
        original_images = np.load('./Data/processed_images.npy').astype(np.float32) / 255.0
        original_labels = np.load('./Data/JAFFE_labels.npy').astype(int)
        id_labels = np.load('./Data/ID_labels.npy').astype(int)
    else:
        raise ValueError("Please choose one datatype from original or processed")

    images, labels = [], []
    for image, label, id_label in zip(original_images, original_labels, id_labels):
        if remove_id_class is not None and id_label == remove_id_class:
            continue
        images.append(image)
        labels.append(label)

    images = np.array(images).reshape(-1, 1, 256, 256)  # Channel first for PyTorch
    labels = np.array(labels)

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, random_state=42)

    # Data augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        # Add other transformations as needed
    ])

    # Create datasets
    train_dataset = CustomDataset(train_images, train_labels, transform=transform)
    test_dataset = CustomDataset(test_images, test_labels, transform=transforms.ToTensor())

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    return train_loader, test_loader

def make_canary_data(data_type='original', id_class=None):
    if data_type == 'original':
        original_images = np.load('./Data/original_images.npy').astype(np.float32) / 255.0
        original_labels = np.load('./Data/JAFFE_labels.npy').astype(int)
        id_labels = np.load('./Data/ID_labels.npy').astype(int)
    elif data_type == 'processed':
        original_images = np.load('./Data/processed_images.npy').astype(np.float32) / 255.0
        original_labels = np.load('./Data/JAFFE_labels.npy').astype(int)
        id_labels = np.load('./Data/ID_labels.npy').astype(int)
    else:
        raise ValueError("Please choose one datatype from original or processed")

    images, labels = [], []
    for image, label, id_label in zip(original_images, original_labels, id_labels):
        if id_class is not None and id_label == id_class:
            images.append(image)
            labels.append(label)

    images = np.array(images).reshape(-1, 1, 256, 256)  # Channel first for PyTorch
    labels = np.array(labels)


    return images,labels

def save_observations(O, O_prime, model_index, canary_index):
    # Save observations to a file
    observations_dir = os.path.join('./models', 'observations')
    if not os.path.exists(observations_dir):
        os.makedirs(observations_dir)

    filename = f'observations_model_{model_index}_canary_{canary_index}.npz'
    filepath = os.path.join(observations_dir, filename)
    np.savez(filepath, O=O, O_prime=O_prime)

def process_observations(O, O_prime):
    # Calculate the mean difference between the losses
    mean_difference = np.mean(np.array(O_prime) - np.array(O))
    return mean_difference

def compute_threshold(O, O_prime):
    # Calculate the decision threshold as the mean of the observations' norms
    threshold = (np.mean([np.linalg.norm(o) for o in O]) +
                 np.mean([np.linalg.norm(o_prime) for o_prime in O_prime])) / 2
    return threshold

def compute_fpr_fnr(O, O_prime, threshold):
    # Compute False Positive Rate and False Negative Rate
    FPR = np.mean([np.linalg.norm(o) >= threshold for o in O])
    FNR = np.mean([np.linalg.norm(o_prime) < threshold for o_prime in O_prime])
    return FPR, FNR


def audit_model(canary, model_without_canary, model_with_canary, criterion):
    canary_x, canary_y = canary

    # Convert canary_x from a NumPy array to a PyTorch tensor
    canary_x = torch.from_numpy(canary_x).float()

    # Convert canary_y from an integer to a PyTorch tensor
    canary_y = torch.tensor([canary_y], dtype=torch.long)  # Wrapping in a list to create a tensor

    # Adding batch dimension to canary_x
    canary_x = torch.unsqueeze(canary_x, 0)

    # Ensure models and data are on the same device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_without_canary.to(device)
    model_with_canary.to(device)
    canary_x, canary_y = canary_x.to(device), canary_y.to(device)

    with torch.no_grad():
        # Predictions for the canary on both models
        pred_without_canary = model_without_canary(canary_x)
        pred_with_canary = model_with_canary(canary_x)

        # Calculate the loss for the canary on both models
        loss_without_canary = criterion(pred_without_canary, canary_y)
        loss_with_canary = criterion(pred_with_canary, canary_y)

    return loss_without_canary.item(), loss_with_canary.item()

def compute_statistics(O, O_prime):
    # Calculate the decision threshold as the midpoint between the means of the two observation sets
    threshold = (np.mean(O) + np.mean(O_prime)) / 2

    # Compute False Positive Rate and False Negative Rate
    FPR = np.mean([o >= threshold for o in O])
    FNR = np.mean([o_prime < threshold for o_prime in O_prime])

    return threshold, FPR, FNR

def compute_empirical_epsilon(FPR, FNR):
    # Assuming delta is close to 0
    delta = 0
    FPHigh = max(FPR, 1e-10)  # Avoid division by zero
    FNHigh = max(FNR, 1e-10)  # Avoid division by zero

    empirical = max(np.log(1 / FPHigh) - FNHigh, np.log(1 / FNHigh) - FPHigh)
    return empirical

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        # Initialize metrics for training
        train_loss = 0
        correct_train = 0
        total_train = 0

        # Training loop
        model.train()
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)
            X_train = X_train.transpose(1, 2)
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += y_train.size(0)
            correct_train += (predicted == y_train).sum().item()

        train_accuracy = 100 * correct_train / total_train

        # Initialize metrics for validation
        val_loss = 0
        correct_val = 0
        total_val = 0

        # Validation loop
        model.eval()
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                X_test = X_test.transpose(1, 2)
                output = model(X_test)
                loss = criterion(output, y_test)

                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_val += y_test.size(0)
                correct_val += (predicted == y_test).sum().item()

        val_accuracy = 100 * correct_val / total_val

        print(f"Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader)}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss / len(test_loader)}, Val Acc: {val_accuracy:.2f}%")

    return model
