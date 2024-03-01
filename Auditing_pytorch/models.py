import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduce=False, num_groups=32):
        super(ResidualBlock, self).__init__()
        stride = 2 if reduce else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)

        self.shortcut = nn.Sequential()
        if reduce:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
                nn.GroupNorm(num_groups, out_channels)
            )

    def forward(self, x):
        y = F.relu(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y)
        return y

class ResNet18(nn.Module):
    def __init__(self, num_classes=7, input_shape=(1, 256, 256), num_groups=32):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3)
        self.gn1 = nn.GroupNorm(num_groups, 64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ResidualBlock(64, 64, reduce=False, num_groups=num_groups)
        self.layer2 = ResidualBlock(64, 128, reduce=True, num_groups=num_groups)
        self.layer3 = ResidualBlock(128, 128, reduce=False, num_groups=num_groups)
        self.layer4 = ResidualBlock(128, 256, reduce=True, num_groups=num_groups)
        self.layer5 = ResidualBlock(256, 256, reduce=False, num_groups=num_groups)
        self.layer6 = ResidualBlock(256, 512, reduce=True, num_groups=num_groups)
        self.layer7 = ResidualBlock(512, 512, reduce=False, num_groups=num_groups)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x
        
def resnet18(num_classes=7, input_shape=(1, 256, 256), max_grad_norm=1.0, batch_size=64, data_loader=None, epochs=25, desired_epsilon=1):
    model = ResNet18(num_classes=num_classes, input_shape=input_shape)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if data_loader is not None:
        sample_size = len(data_loader.dataset)
        sample_rate = batch_size / sample_size
        target_delta = 1 / sample_size
    else:
        raise ValueError("DataLoader is required for differential privacy setup")

    privacy_engine = PrivacyEngine()
    model, optimizer, _ = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        target_epsilon=desired_epsilon,
        target_delta=0.0002,
        max_grad_norm=max_grad_norm,
        epochs=epochs
    )

    return model, optimizer, loss_fn, privacy_engine
