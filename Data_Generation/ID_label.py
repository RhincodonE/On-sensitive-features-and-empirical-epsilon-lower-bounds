import os
import numpy as np
import cv2

# Directory containing the image files
image_directory = '/Users/rhincodon/Learning/Phd_Projects/Explain/Face_Express/Dataset/JAFFE'

# Create lists to store images and labels
images = []
labels = []

# Loop through all the image files in the directory
for filename in os.listdir(image_directory):
    if filename.endswith('.tiff'):
        parts = filename.split('.')
        if len(parts) >= 2:
            class_name = parts[0]
            class_name = ''.join([c for c in class_name if not c.isdigit()])  # Remove numbers
            labels.append(class_name)

            image_path = os.path.join(image_directory, filename)
            image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
            images.append(image)

# Convert the lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Save images and labels to numpy files

label_mapping={'KA':0,'KL':1, 'KM':2, 'KR':3, 'MK':4, 'NA':5, 'NM':6,'TM':7, 'UY':8, 'YM':9}

numeric_labels = np.array([label_mapping[label] for label in labels])

# Print the number of unique numeric labels and the numeric labels themselves
unique_numeric_labels = np.unique(numeric_labels)
print("Dataset shape:", images.shape)
print("Number of unique numeric labels:", len(unique_numeric_labels))
print("Unique numeric labels:", unique_numeric_labels)

# Save numeric labels to a numpy file
np.save('ID_labels.npy', numeric_labels)
np.save('ID_images.npy', images)

print("Images and labels saved to numpy files.")
