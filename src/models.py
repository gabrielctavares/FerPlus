import torch.nn as nn

from torchvision import models

def build_model(num_classes, model_name):
    model = globals()[model_name]
    return model(num_classes)

class VGG13(nn.Module):
    learning_rate = 0.05
    input_width = 64
    input_height = 64
    input_channels = 1

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # bloco 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
            # bloco 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
            # bloco 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
            # bloco 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
        )

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


class ResNet18(nn.Module):
    """
    ResNet18 adaptada para FER+ 1x64x64 inputs e num_classes finais.
    """
    learning_rate = 0.01
    input_width = 64
    input_height = 64
    input_channels = 1
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)        
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
    

    
# implementar  VGG16, VGG19, DenseNet, EfficientNet, ConvNext,  xception e inception
# entrada com somente 1 canal e saida = num_classes

#Loss ta explodindo 
class VGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
        self.model.classifier[6] = nn.Linear(4096, num_classes)


        print(self.model)

    def forward(self, x):
        return self.model(x)

#Loss ta explodindo 
class VGG19(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.classifier[6] = nn.Linear(4096, num_classes)        
        print(self.model)
    def forward(self, x):
        return self.model(x)

class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.model.classifier = nn.Linear(1024, num_classes)
        print(self.model)
    def forward(self, x):
        return self.model(x)


class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)        
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier[1] = nn.Linear(1280, num_classes)
        print(self.model)

    def forward(self, x):
        return self.model(x)

class ConvNext(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.model.features[0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4), padding=(0, 0), bias=False)
        self.model.classifier[2] = nn.Linear(768, num_classes)
        print(self.model)
    def forward(self, x):
        return self.model(x)