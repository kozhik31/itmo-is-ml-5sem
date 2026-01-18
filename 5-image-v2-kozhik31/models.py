import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import decode_image
import torch
from torchvision import models
from constants import *


class CaloriesCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128,256),

            nn.AdaptiveAvgPool2d(7)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*7*7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

    def get_embedding(self, x):
        x = self.features(x)
        x = self.regressor[:-1](x)
        return x


class CaloriesResNet(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()

        self.backbone = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT
        )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # удаляем последний слой

        self.embedding = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Linear(64, 1)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        x = self.output(x)
        return x

    def get_embedding(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return x

class ImgDataset(Dataset):
    def __init__(self, paths, targets, transforms):
        self.paths = paths
        self.targets = targets
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = decode_image(path)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        image = self.transforms(image)

        return image, target