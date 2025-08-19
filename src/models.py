
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_model(num_classes, model_name):
    model = globals()[model_name]
    return model(num_classes)

class VGG13(nn.Module):
    """
    VGG13-like model as in the CNTK version, tweaked for 1x64x64 FER+ inputs.
    Layout mirrors models.py exactly: 
      (2x conv64) + pool + drop
      (2x conv128) + pool + drop
      (3x conv256) + pool + drop
      (3x conv256) + pool + drop
      FC1024 + ReLU + Drop
      FC1024 + ReLU + Drop
      FC num_classes (no activation)
    """
    learning_rate = 0.05
    input_width = 64
    input_height = 64
    input_channels = 1

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
            # block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
            # block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
            # block 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
        )
        # compute the flattened size after 4 pools on 64x64 -> 4x4 spatial with 256 channels
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
