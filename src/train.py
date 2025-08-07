import torch
from torch import optim
from torch.utils.data import DataLoader

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
import json

from definitions import logging

from data_augmentation import DataAugmentation
from trainer import Trainer
from balancer import ClassBalancer

from ferplus import FERPlusDataset, FERPlusParameters
from models import build_model  



from definitions import logging, device, emotion_table, emotion_names

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
        preload_data=False  # Usa a versão otimizada
    )
    
    params_val_test = FERPlusParameters(
        target_size=len(emotion_table), 
        width=64, height=64, 
        training_mode=mode, 
        deterministic=True, 
        shuffle=False,
        num_workers=4,
        preload_data=False
    )
    
    # Sistema de augmentation
    augmentation = DataAugmentation(mode, augmentation_intensity)
    
    # Datasets
    logging.info("Criando datasets...")
    datasets = {}
    for split in paths:
        is_train = (split == 'train')
        params = params_train if is_train else params_val_test
        transform = augmentation.get_train_transforms(params.height, params.width) if is_train else augmentation.get_val_transforms(params.height, params.width)
        
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
    train_sampler = None # balancer.create_weighted_sampler(datasets['train'])
    
    # Pesos das classes para loss function
    class_weights = balancer.compute_class_weights(datasets['train'].per_emotion_count)
    logging.info(f"Class weights: {class_weights}")
    
    # DataLoaders otimizados
    dataloaders = {}
    for split in paths:
        shuffle = (split == 'train')
        sampler = train_sampler if split == 'train' else None
        
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=bs,
            shuffle=shuffle,
            #sampler=sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
    
    # Modelo e otimizador
    model = build_model(len(emotion_table), model_name).to(device)
    
    # Otimizador com configurações melhoradas
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=1e-3,
    #     weight_decay=1e-4,
    #     momentum=0.9
    # )
    lr = getattr(model, "learning_rate", 0.01)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

   
    # Trainer
    trainer = Trainer(model_name, mode, len(emotion_table))
    
    # Loop de treinamento
    logging.info("Iniciando treinamento...")
    best_model_path = f"{model_name}_{mode}.pth"
    
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