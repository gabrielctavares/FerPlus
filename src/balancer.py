from typing import Optional
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from ferplus import FERPlusDataset

class ClassBalancer:
    """
    Sistema de balanceamento de classes com múltiplas estratégias
    """
    def __init__(self, strategy: str = "weighted_sampling"):
        self.strategy = strategy
        
    def compute_class_weights(self, class_counts: np.ndarray) -> torch.Tensor:
        """
        Computa pesos das classes usando diferentes estratégias
        
        Estratégias:
        - 'inverse_frequency': 1/freq
        - 'sqrt_inverse': 1/sqrt(freq)
        - 'log_inverse': 1/log(1+freq)
        - 'focal': Para focal loss
        """
        total_samples = class_counts.sum()
        n_classes = len(class_counts)
        
        if self.strategy == "inverse_frequency":
            # Peso = Total_samples / (n_classes * class_count)
            weights = total_samples / (n_classes * (class_counts + 1e-6))
        
        elif self.strategy == "sqrt_inverse":
            # Suaviza o balanceamento usando raiz quadrada
            frequencies = class_counts / total_samples
            weights = 1.0 / (np.sqrt(frequencies) + 1e-6)
        
        elif self.strategy == "log_inverse":
            # Balanceamento ainda mais suave usando log
            frequencies = class_counts / total_samples
            weights = 1.0 / (np.log(frequencies + 1) + 1e-6)
        
        elif self.strategy == "focal":
            # Para uso com focal loss
            frequencies = class_counts / total_samples
            weights = (1 - frequencies) / frequencies
        
        else:
            # Default: inverse frequency
            weights = total_samples / (n_classes * (class_counts + 1e-6))
        
        # Normaliza os pesos
        weights = weights / weights.sum() * n_classes
        
        return None #torch.tensor(weights, dtype=torch.float32)
    
    def create_weighted_sampler(self, dataset: FERPlusDataset) -> Optional[WeightedRandomSampler]:
        """Cria um sampler balanceado para o DataLoader"""
        if self.strategy == "none":
            return None
            
        # Obtém contagens das classes
        class_counts = dataset.per_emotion_count
        class_weights = self.compute_class_weights(class_counts)
        
        # Calcula peso para cada amostra
        sample_weights = []
        for _, emotion_labels, _ in dataset.data:
            # Encontra a classe majoritária
            class_idx = np.argmax(emotion_labels)
            sample_weights.append(class_weights[class_idx].item())
        
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
