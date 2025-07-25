# models.py
import torch
import torch.nn as nn

def build_model(num_classes, model_name):
    """
    Factory function para instanciar o modelo.
    """
    model = globals()[model_name]
    return model(num_classes)

class VGG13(nn.Module):
    """
    VGG13-like model adaptado para FER+.
    """
    @property
    def learning_rate(self):
        return 0.01
        
    @property
    def input_width(self):
        return 64
        
    @property
    def input_height(self):
        return 64
    
    def __init__(self, num_classes):
        super(VGG13, self).__init__()
        # hiperparâmetros herdados
        
        self.input_channels = 1

        # blocos convolucionais
        self.features = nn.Sequential(
            # bloco 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # bloco 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # bloco 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # bloco 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        # classificadores fully-connected
        # saída de 64x64 após 4 pooling 2x2 => 4x4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
