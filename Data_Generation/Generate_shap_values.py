import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms,models
from torch.utils.data import DataLoader, Subset ,TensorDataset
import shap
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes)
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut_conv:
            shortcut = self.shortcut_bn(self.shortcut_conv(x))
        else:
            shortcut = x
        out = out + shortcut
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Input channels set to 1 (grayscale)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)  # Use nn.Sequential for the layers

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

# Define network architecture
device = torch.device("cuda")


# Load data
images = np.load('ID_images.npy')
labels = np.load('ID_labels.npy')
print(images[0])

# Convert labels to integers (if they are not already)
labels = labels.astype(int)

# Convert images to float32 and normalize to [0, 1]
images = images.astype(np.float32) / 255.0

# Convert the numpy arrays to PyTorch tensors
images_tensor = torch.tensor(images)
labels_tensor = torch.tensor(labels)

# Define the dataset using TensorDataset
dataset = TensorDataset(images_tensor, labels_tensor)

# Define batch sizes for training and testing
batch_size = 3

# Split the dataset into training and testing sets
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = ResNet18().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
criterion = nn.CrossEntropyLoss()
# Training loop
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data.unsqueeze(1))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:

            data = data.to(device)
            target = target.to(device)
            output = model(data.unsqueeze(1))
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(30):
    train(epoch)
    test()

# Load background data
print('Loading background data...')
background_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
background_images = next(iter(background_loader))[0].unsqueeze(1).to(device)

# Initialize the explainer with the model and the background data
e = shap.DeepExplainer(model, background_images)

# Define the number of images to sample per class


# Initialize lists to store sampled images and labels
num_images_per_class = 21
num_classes = 10

# Initialize lists to store sampled images and labels
sampled_images = []
sampled_labels = [0] * num_classes


# Loop over the training dataset to sample images from each class
print('Loading explanation data...')
for image, label in train_dataset:
    if sampled_labels[label] < num_images_per_class:
        sampled_images.append(image)
        sampled_labels[label] += 1

    # Check if all classes have reached the desired number of samples
    if all(count == num_images_per_class for count in sampled_labels):
        break

# Convert the sampled images and labels to tensors
sampled_images_tensor = torch.stack(sampled_images)
sampled_labels_tensor = torch.tensor(sampled_labels * num_classes)

print(f"Sampled {len(sampled_images_tensor)} images from {num_classes} classes")

# Calculate SHAP values for the sampled images
# Calculate SHAP values for the sampled images
shap_values = []
batch_size = batch_size  # Adjust this value based on your GPU memory
num_batches = len(sampled_images_tensor) // batch_size

with tqdm(total=len(sampled_images_tensor), desc="Calculating SHAP values") as pbar:
    for i in range(num_batches + 1):
        start = i * batch_size
        end = start + batch_size
        images_batch = sampled_images_tensor[start:end].unsqueeze(1).to(device)
        for image in images_batch:
            print(image.shape)
            shap_value = e.shap_values(image.unsqueeze(0).clone())
            shap_values.append(shap_value)
            pbar.update(1)


print(np.array(shap_values).shape)
shap_values = np.array(shap_values)
shap_values = np.squeeze(shap_values, axis=2)
print(shap_values.shape)
shap_values_reshaped = np.transpose(shap_values, (1, 0, 2, 3, 4))
print(shap_values_reshaped.shape)

# Save the SHAP values for each class
for class_idx, shap_values_class in enumerate(shap_values_reshaped):
    np.save(f'./Values/shap_values_id_{class_idx}.npy', shap_values_class)
