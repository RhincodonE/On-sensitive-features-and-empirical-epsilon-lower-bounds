import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from ortools.linear_solver import pywraplp
from tqdm import tqdm
# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process images based on shap values')
parser.add_argument('--delta', type=float, default=0.2)
parser.add_argument('--n', type=int, default=1000,
                    help='number of elements to draw')
parser.add_argument('--threshold', type=float, default=0.007,
                    help='threshold value for pixel modification')

args = parser.parse_args()
n = args.n
delta = args.delta
threshold = args.threshold


# Define the list of shap values files for each class
npy_files = [
    './Values/shap_values_class_0.npy',
    './Values/shap_values_class_1.npy',
    './Values/shap_values_class_2.npy',
    './Values/shap_values_class_3.npy',
    './Values/shap_values_class_4.npy',
    './Values/shap_values_class_5.npy',
    './Values/shap_values_class_6.npy'
]

npy_files_id = [
    './Values/shap_values_id_0.npy',
    './Values/shap_values_id_1.npy',
    './Values/shap_values_id_2.npy',
    './Values/shap_values_id_3.npy',
    './Values/shap_values_id_4.npy',
    './Values/shap_values_id_5.npy',
    './Values/shap_values_id_6.npy',
    './Values/shap_values_id_7.npy',
    './Values/shap_values_id_8.npy',
    './Values/shap_values_id_9.npy'
]

total_sum = None

P = []

for file_name in npy_files_id:
    # Load .npy file
    shap_values = np.load(file_name)

    # Check if the number of arrays to draw (n) is greater than the number of arrays in the npy file
    if n > shap_values.shape[0]:
        print(f"Warning: {file_name} contains fewer than {n} arrays. Drawing all available arrays.")

    # Randomly select n arrays from the loaded npy file (or all arrays if n exceeds the available number of arrays)
    selected_arrays = shap_values[np.random.choice(shap_values.shape[0], size=min(n, shap_values.shape[0]), replace=False)]

    # Calculate the average value element-wise for the selected arrays
    average_values = np.mean(selected_arrays, axis=0)

    # Reshape the average values array to (28, 28)
    average_values = average_values.reshape((256, 256))

    # Set negative values to zero
    average_values[average_values < 0] = 0

    # If total_sum is still None, initialize it with the shape of the loaded data
    P.append(average_values.ravel())


for file_name in npy_files:
    # Load .npy file
    shap_values = np.load(file_name)

    # Check if the number of arrays to draw (n) is greater than the number of arrays in the npy file
    if n > shap_values.shape[0]:
        print(f"Warning: {file_name} contains fewer than {n} arrays. Drawing all available arrays.")

    # Randomly select n arrays from the loaded npy file (or all arrays if n exceeds the available number of arrays)
    selected_arrays = shap_values[np.random.choice(shap_values.shape[0], size=min(n, shap_values.shape[0]), replace=False)]

    # Calculate the average value element-wise for the selected arrays
    average_values = np.mean(selected_arrays, axis=0)

    # Reshape the average values array to (28, 28)
    average_values = average_values.reshape((256, 256))

    average_values[average_values < 0] = 0

    # If total_sum is still None, initialize it with the shape of the loaded data
    if total_sum is None:
        total_sum = np.zeros_like(average_values)

    # Add data to total_sum
    total_sum += average_values


# Compute the mean value
mean_value_ut = total_sum / len(npy_files)

original_shape = mean_value_ut.shape
U = mean_value_ut.ravel()

print(U.shape)
m_solutions = []

# Create a solver
solver = pywraplp.Solver.CreateSolver('GLOP')



# For each u in U, solve the optimization problem

i=1
for p in P:
    print('solving ' + str(i))
    i=i+1
    if not solver:
        raise Exception("Solver not found!")

    # Define our variables - 0 <= m_i <= 1
    m = []
    for j in tqdm(range(len(U)), desc="Creating variables", ncols=100):
        m.append(solver.BoolVar('m[%i]' % j))

    # Define the constraints: m^T * u > (1-delta) * sum(u)
    constraint_expr = sum([m[j] * p[j] for j in range(len(p))])
    solver.Add(constraint_expr >= (1-delta) * np.sum(p))

    # Define the objective function: m^T * P
    solver.Minimize(sum([m[j] * U[j] for j in range(len(U))]))

    # Solve the problem and obtain the solution


    status = solver.Solve()


    if status == pywraplp.Solver.OPTIMAL:
        m_solution = [int(m[j].solution_value()) for j in range(len(m))]
        m_image = np.array(m_solution).reshape(original_shape)
        m_solutions.append(m_image)
        print(np.min(m_image))
        print(np.max(m_image))
    else:
        print(p.shape)
        for j in range(p.shape[0]):
                if p[j] < threshold:
                    p[j] = 0
                else:
                    p[j] = 1
        m_image = np.array(p).reshape(original_shape)
        m_solutions.append(m_image)
        print(f"No solution found for current, generate mask through threshold.")
# Load JAFFE_images and JAFFE_labels
JAFFE_images = np.load('./Data/original_images.npy')
JAFFE_labels = np.load('./Data/JAFFE_labels.npy')

# Validate the dimensions
if len(JAFFE_images) != len(JAFFE_labels):
    raise ValueError("The number of images and labels do not match!")

# Create a list to store the masked images
masked_images = []

# Apply the masks
for image, label in zip(JAFFE_images, JAFFE_labels):
    if label >= len(m_solutions) or m_solutions[label] is None:
        # No mask for this label or there was no solution for the mask
        masked_images.append(image)
        continue

    # Apply the mask
    masked_img = image * m_solutions[label]

    # Append the masked image to the list
    masked_images.append(masked_img)

# Convert the list to a numpy array
masked_images = np.array(masked_images)

# Save the masked images
np.save('./Data/Optimized_Masked_images.npy', masked_images)

for idx, m_image in enumerate(m_solutions):
        np.save(f"./Masks/m{idx+1}.npy", m_image)
