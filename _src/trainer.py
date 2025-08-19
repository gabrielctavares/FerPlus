from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from definitions import device, emotion_table, emotion_names, logging

class Trainer:
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
            
            weights = class_weights.to(device) if class_weights is not None else None
            loss = self.compute_loss(outputs, y, weights)
            
            loss.backward()
            
            # Gradient clipping para estabilidade
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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

