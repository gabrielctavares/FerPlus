import torch
import os
import time
import logging
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
import json

from ferplus import FERPlusDataset, FERPlusParameters
from models import build_model  
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

emotion_table = {
    'neutral': 0, 'happiness': 1, 'surprise': 2, 'sadness': 3,
    'anger': 4, 'disgust': 5, 'fear': 6, 'contempt': 7
}

emotion_names = list(emotion_table.keys())

class AdvancedDataAugmentation:
    """
    Sistema avançado de data augmentation específico para expressões faciais
    """
    def __init__(self, training_mode: str = "majority", intensity: str = "medium"):
        self.training_mode = training_mode
        self.intensity = intensity
        
        # Define intensidade da augmentation
        intensity_configs = {
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
            transforms.Lambda(lambda x: self._add_gaussian_noise(x, prob=0.1))
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
        
        return torch.tensor(weights, dtype=torch.float32)
    
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

class ImprovedTrainer:
    """
    Classe de treinamento aprimorada com métricas avançadas
    """
    def __init__(self, model_name: str, training_mode: str, num_classes: int = 8):
        self.model_name = model_name
        self.training_mode = training_mode
        self.num_classes = num_classes
        
        # Configurações
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=f"runs/{model_name}_{training_mode}")
    
    def compute_loss(self, prediction_logits: torch.Tensor, target: torch.Tensor, 
                    class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Função de loss melhorada com suporte a class weights e diferentes modos
        """
        if self.training_mode in ['majority', 'probability', 'crossentropy']:
            labels = torch.argmax(target, dim=1)
            
            # Cross entropy com class weights e label smoothing
            loss = F.cross_entropy(
                prediction_logits, 
                labels, 
                weight=class_weights,
                label_smoothing=0.1
            )
            
        elif self.training_mode == 'multi_target':
            # Multi-label loss otimizado
            pred_probs = F.softmax(prediction_logits, dim=1)
            
            # BCE com logits para multi-label
            loss = F.binary_cross_entropy_with_logits(
                prediction_logits,
                target,
                pos_weight=class_weights
            )
            
        else:
            raise ValueError(f"Modo de treinamento inválido: {self.training_mode}")
        
        return loss
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Computa métricas detalhadas"""
        if targets.dim() > 1:
            targets = targets.argmax(dim=1)
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=1)
        
        # Converte para numpy para sklearn
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Accuracy
        accuracy = (preds_np == targets_np).mean()
        
        # Per-class accuracy
        per_class_acc = {}
        for i, emotion in enumerate(emotion_names):
            mask = targets_np == i
            if mask.sum() > 0:
                per_class_acc[emotion] = (preds_np[mask] == targets_np[mask]).mean()
            else:
                per_class_acc[emotion] = 0.0
        
        return {
            'accuracy': accuracy,
            'per_class_accuracy': per_class_acc
        }
    
    def train_epoch(self, model: torch.nn.Module, dataloader: DataLoader, 
                   optimizer: torch.optim.Optimizer, epoch: int,
                   class_weights: Optional[torch.Tensor] = None) -> Tuple[float, float]:
        """Treina uma época"""
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        
        progress_bar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            
            # Move class_weights para o device se necessário
            weights = class_weights.to(device) if class_weights is not None else None
            loss = self.compute_loss(outputs, y, weights)
            
            loss.backward()
            
            # Gradient clipping para estabilidade
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Métricas
            preds = outputs.argmax(dim=1)
            trues = y.argmax(dim=1) if y.dim() > 1 else y
            
            running_loss += loss.item() * x.size(0)
            running_correct += (preds == trues).sum().item()
            running_total += x.size(0)
            
            # Atualiza progress bar
            current_acc = running_correct / running_total
            progress_bar.set_postfix({
                'Loss': f'{running_loss/running_total:.4f}',
                'Acc': f'{current_acc:.4f}'
            })
        
        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total
        
        return epoch_loss, epoch_acc
    
    def validate(self, model: torch.nn.Module, dataloader: DataLoader, 
                class_weights: Optional[torch.Tensor] = None) -> Tuple[float, float, Dict]:
        """Valida o modelo"""
        model.eval()
        running_loss, running_correct, running_total = 0.0, 0, 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Validation"):
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                
                weights = class_weights.to(device) if class_weights is not None else None
                loss = self.compute_loss(outputs, y, weights)
                
                preds = outputs.argmax(dim=1)
                trues = y.argmax(dim=1) if y.dim() > 1 else y
                
                running_loss += loss.item() * x.size(0)
                running_correct += (preds == trues).sum().item()
                running_total += x.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(trues.cpu().numpy())
        
        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total
        
        # Métricas detalhadas
        detailed_metrics = self.compute_metrics(
            torch.tensor(all_preds), 
            torch.tensor(all_targets)
        )
        
        return epoch_loss, epoch_acc, detailed_metrics
    
    def save_confusion_matrix(self, y_true: List[int], y_pred: List[int], epoch: int):
        """Salva matriz de confusão"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=emotion_names, yticklabels=emotion_names)
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Salva no TensorBoard
        self.writer.add_figure(f'Confusion_Matrix/Epoch_{epoch}', plt.gcf(), epoch)
        plt.close()
    
    def log_metrics(self, epoch: int, train_loss: float, train_acc: float,
                   val_loss: float, val_acc: float, detailed_metrics: Dict):
        """Registra métricas no TensorBoard"""
        # Métricas principais
        self.writer.add_scalars('Loss', {
            'Train': train_loss,
            'Validation': val_loss
        }, epoch)
        
        self.writer.add_scalars('Accuracy', {
            'Train': train_acc,
            'Validation': val_acc
        }, epoch)
        
        # Accuracy per classe
        for emotion, acc in detailed_metrics['per_class_accuracy'].items():
            self.writer.add_scalar(f'Per_Class_Accuracy/{emotion}', acc, epoch)
        
        # Histórico
        self.training_history['train_loss'].append(train_loss)
        self.training_history['train_acc'].append(train_acc)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['val_acc'].append(val_acc)

def main(base_folder: str, mode: str = 'majority', model_name: str = 'VGG13', 
         epochs: int = 3, bs: int = 64, balance_strategy: str = "weighted_sampling",
         augmentation_intensity: str = "medium"):
    
    # Caminhos dos dados
    paths = {
        'train': 'FER2013Train',
        'valid': 'FER2013Valid',
        'test': 'FER2013Test'
    }
    
    # Parâmetros otimizados
    params_train = FERPlusParameters(
        target_size=len(emotion_table), 
        width=64, height=64, 
        training_mode=mode, 
        deterministic=False, 
        shuffle=True,
        num_workers=4,
        preload_data=True  # Usa a versão otimizada
    )
    
    params_val_test = FERPlusParameters(
        target_size=len(emotion_table), 
        width=64, height=64, 
        training_mode=mode, 
        deterministic=True, 
        shuffle=False,
        num_workers=4,
        preload_data=True
    )
    
    # Sistema de augmentation
    augmentation = AdvancedDataAugmentation(mode, augmentation_intensity)
    
    # Datasets
    logging.info("Criando datasets...")
    datasets = {}
    for split in paths:
        is_train = (split == 'train')
        params = params_train if is_train else params_val_test
        transform = augmentation.get_train_transforms(params.height, params.width) if is_train \
                   else augmentation.get_val_transforms(params.height, params.width)
        
        datasets[split] = FERPlusDataset(
            base_folder=base_folder,
            sub_folders=[paths[split]],
            label_file_name="label.csv",
            parameters=params,
            transform=transform
        )
    
    # Exibe distribuição das classes
    for split, dataset in datasets.items():
        logging.info(f"\n=== {split.upper()} DATASET ===")
        distribution = dataset.get_emotion_distribution()
        for emotion, stats in distribution.items():
            logging.info(f"{emotion}: {stats['count']} samples ({stats['percentage']:.1f}%)")
    
    # Sistema de balanceamento
    balancer = ClassBalancer(balance_strategy)
    train_sampler = balancer.create_weighted_sampler(datasets['train'])
    
    # Pesos das classes para loss function
    class_weights = balancer.compute_class_weights(datasets['train'].per_emotion_count)
    logging.info(f"Class weights: {class_weights}")
    
    # DataLoaders otimizados
    dataloaders = {}
    for split in paths:
        shuffle = (split == 'train' and train_sampler is None)
        sampler = train_sampler if split == 'train' else None
        
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=bs,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
    
    # Modelo e otimizador
    model = build_model(len(emotion_table), model_name).to(device)
    
    # Otimizador com configurações melhoradas
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Trainer
    trainer = ImprovedTrainer(model_name, mode, len(emotion_table))
    
    # Loop de treinamento
    logging.info("Iniciando treinamento...")
    best_model_path = f"{model_name}_{mode}_balanced.pth"
    
    for epoch in range(1, epochs + 1):
        logging.info(f"\n=== EPOCH {epoch}/{epochs} ===")
        
        # Treinamento
        train_loss, train_acc = trainer.train_epoch(
            model, dataloaders['train'], optimizer, epoch, class_weights
        )
        
        # Validação
        val_loss, val_acc, detailed_metrics = trainer.validate(
            model, dataloaders['valid'], class_weights
        )
        
        # Scheduler step
        scheduler.step()
        
        # Log métricas
        trainer.log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, detailed_metrics)
        
        # Log detalhado
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Per-class accuracy
        for emotion, acc in detailed_metrics['per_class_accuracy'].items():
            logging.info(f"  {emotion}: {acc:.4f}")
        
        # Salva melhor modelo
        if val_acc > trainer.best_val_acc:
            trainer.best_val_acc = val_acc
            trainer.best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'training_history': trainer.training_history
            }, best_model_path)
            logging.info(f"🚀 Novo melhor modelo salvo! Val Acc: {val_acc:.4f}")
    
    # Teste final
    logging.info("\n=== TESTE FINAL ===")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_metrics = trainer.validate(
        model, dataloaders['test'], class_weights
    )
    
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    logging.info("Per-class Test Accuracy:")
    for emotion, acc in test_metrics['per_class_accuracy'].items():
        logging.info(f"  {emotion}: {acc:.4f}")
    
    # Salva relatório final
    final_report = {
        'model_name': model_name,
        'training_mode': mode,
        'balance_strategy': balance_strategy,
        'augmentation_intensity': augmentation_intensity,
        'best_epoch': trainer.best_epoch,
        'best_val_acc': trainer.best_val_acc,
        'test_acc': test_acc,
        'test_per_class': test_metrics['per_class_accuracy'],
        'training_history': trainer.training_history
    }
    
    with open(f"training_report_{model_name}_{mode}.json", 'w') as f:
        json.dump(final_report, f, indent=2)
    
    trainer.writer.close()
    logging.info(f"Treinamento concluído! Melhor modelo: {best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento FER+ Otimizado")
    parser.add_argument("-d", "--base_folder", required=True, help="Pasta base do dataset")
    parser.add_argument("-m", "--training_mode", default="majority", 
                       choices=["majority", "probability", "crossentropy", "multi_target"])
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("--model", default="VGG13", help="Nome do modelo")
    parser.add_argument("--balance", default="weighted_sampling", 
                       choices=["none", "inverse_frequency", "sqrt_inverse", "weighted_sampling"])
    parser.add_argument("--augmentation", default="medium",
                       choices=["light", "medium", "heavy"])
    
    args = parser.parse_args()
    
    main(
        base_folder=args.base_folder,
        mode=args.training_mode,
        model_name=args.model,
        epochs=args.epochs,
        bs=args.batch_size,
        balance_strategy=args.balance,
        augmentation_intensity=args.augmentation
    )