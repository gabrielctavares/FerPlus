import torch.nn as nn

from torchvision import models
import torch

def build_model(num_classes, model_name):
    model = globals()[model_name]
    return model(num_classes)



import torch
import torch.nn.functional as F
from torch import nn

class VGG16(nn.Module):
    learning_rate = 0.01
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)

class VGG19(nn.Module):
    learning_rate = 0.01
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
        self.model.classifier[6] = nn.Linear(4096, num_classes)        
        
    def forward(self, x):
        return self.model(x)

class ResNet18(nn.Module):
    learning_rate = 0.01
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) 
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.model.maxpool = nn.Identity()
        
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
        
        
class DenseNet(nn.Module):
    learning_rate = 0.005
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
   
        self.model.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        return self.model(x)


class EfficientNet(nn.Module):
    learning_rate = 0.001
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)   
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
            
        self.model.classifier[1] = nn.Linear(1280, num_classes)
        

    def forward(self, x):
        return self.model(x)

class ConvNext(nn.Module):
    learning_rate = 0.001
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)        
        self.model.features[0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=1, padding=1, bias=False)
        
            
        self.model.classifier[2] = nn.Linear(768, num_classes)
        
    def forward(self, x):
        return self.model(x)