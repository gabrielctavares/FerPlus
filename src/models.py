import os
import torch
import torch.nn as nn

from torchvision import models

def build_model(num_classes, model_name, ferplus=True, checkpoint_path=None):
    model_class = globals()[model_name]
    model = model_class(num_classes, ferplus)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state"]
        # new_state_dict = {
        #     k.replace("model.", "", 1): v
        #     for k, v in state_dict.items()
        # }
        model.load_state_dict(state_dict)
       
    return model

        
class VGG16(nn.Module):
    learning_rate = 0.01
    def __init__(self, num_classes, ferplus):
        super().__init__()
        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        if ferplus:
            self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)

class VGG19(nn.Module):
    learning_rate = 0.01
    def __init__(self, num_classes, ferplus):
        super().__init__()
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        if ferplus:
            self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.classifier[6] = nn.Linear(4096, num_classes)        
        
    def forward(self, x):
        return self.model(x)

class ResNet18(nn.Module):
    learning_rate = 0.01
    # input_width = 64
    # input_height = 64
    # input_channels = 1
    def __init__(self, num_classes, ferplus):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)        
        if ferplus:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class DenseNet(nn.Module):
    learning_rate = 0.005
    def __init__(self, num_classes, ferplus):
        super().__init__()
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        if ferplus:
            self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.model.classifier = nn.Linear(1024, num_classes)

        # # conv inicial treinável
        # for p in self.model.features[0].parameters():
        #     p.requires_grad = True
        
    def forward(self, x):
        return self.model(x)


class EfficientNet(nn.Module):
    learning_rate = 0.001
    def __init__(self, num_classes, ferplus):
        super().__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)        
        if ferplus:
            self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier[1] = nn.Linear(1280, num_classes)

        # conv inicial deve ser treinável
        # for p in self.model.features[0][0].parameters():
        #     p.requires_grad = True

    def forward(self, x):
        return self.model(x)

class ConvNext(nn.Module):
    learning_rate = 0.001
    def __init__(self, num_classes, ferplus):
        super().__init__()
        self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        if ferplus:
            self.model.features[0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4), padding=(0, 0), bias=False)
        self.model.classifier[2] = nn.Linear(768, num_classes)

        # conv inicial treinável
        # for p in self.model.features[0].parameters():
        #     p.requires_grad = True
        
    def forward(self, x):
        return self.model(x)