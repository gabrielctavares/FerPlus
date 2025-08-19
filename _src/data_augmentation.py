import torch
from torchvision import transforms

class DataAugmentation:
    """
    Sistema avançado de data augmentation específico para expressões faciais
    """
    def __init__(self, training_mode: str = "majority", intensity: str = "medium"):
        self.training_mode = training_mode
        self.intensity = intensity
        
        # Define intensidade da augmentation
        intensity_configs = {
            "none": {
                "rotation_range": 0,
                "zoom_range": 0,
                "shift_range": 0,
                "brightness_range": 0,
                "contrast_range": 0,
                "flip_prob": 0
            },
            "light": {
                "rotation_range": 10,
                "zoom_range": 0.1,
                "shift_range": 0.05,
                "brightness_range": 0.1,
                "contrast_range": 0.1,
                "flip_prob": 0.3
            },
            "medium": {
                "rotation_range": 15,
                "zoom_range": 0.15,
                "shift_range": 0.08,
                "brightness_range": 0.2,
                "contrast_range": 0.2,
                "flip_prob": 0.5
            },
            "heavy": {
                "rotation_range": 20,
                "zoom_range": 0.2,
                "shift_range": 0.1,
                "brightness_range": 0.3,
                "contrast_range": 0.3,
                "flip_prob": 0.6
            }
        }
        
        self.config = intensity_configs.get(intensity, intensity_configs["medium"])
    
    def get_train_transforms(self, height: int, width: int) -> transforms.Compose:
        """Retorna transformações de treinamento otimizadas"""
        if self.intensity == "none":
            return self.get_val_transforms(height, width)

        return transforms.Compose([
            # Redimensionamento inteligente
            transforms.Resize((int(height * 1.1), int(width * 1.1))),
            
            # Crop aleatório para simular variações de enquadramento
            transforms.RandomResizedCrop(
                size=(height, width),
                scale=(1.0 - self.config["zoom_range"], 1.0 + self.config["zoom_range"]),
                ratio=(0.9, 1.1)
            ),
            
            # Rotação e transformações geométricas
            transforms.RandomAffine(
                degrees=self.config["rotation_range"],
                translate=(self.config["shift_range"], self.config["shift_range"]),
                scale=(0.95, 1.05),
                shear=5
            ),
            
            # Flip horizontal (cuidado com assimetrias faciais)
            transforms.RandomHorizontalFlip(p=self.config["flip_prob"]),
            
            # Transformações de cor/intensidade
            transforms.ColorJitter(
                brightness=self.config["brightness_range"],
                contrast=self.config["contrast_range"],
                saturation=0,  # Mantém grayscale
                hue=0
            ),
            
            # Conversão para tensor
            transforms.ToTensor(),
            
            # Normalização otimizada para faces
            transforms.Normalize(mean=[0.485], std=[0.229]),  # ImageNet single channel
            
            # Adiciona ruído gaussiano ocasionalmente
            #transforms.Lambda(lambda x: self._add_gaussian_noise(x, prob=0.1))
        ])
    
    def get_val_transforms(self, height: int, width: int) -> transforms.Compose:
        """Retorna transformações de validação/teste"""
        return transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    def _add_gaussian_noise(self, tensor: torch.Tensor, prob: float = 0.1) -> torch.Tensor:
        """Adiciona ruído gaussiano com probabilidade prob"""
        if torch.rand(1).item() < prob:
            noise = torch.randn_like(tensor) * 0.01
            tensor = torch.clamp(tensor + noise, 0, 1)
        return tensor
