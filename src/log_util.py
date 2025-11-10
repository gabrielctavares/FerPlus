import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from sklearn.metrics import ConfusionMatrixDisplay


def save_results_to_excel(file_path, row_data):
    import pandas as pd
    try:
        df = pd.read_excel(file_path, sheet_name="resultados")
    except FileNotFoundError:
        df = pd.DataFrame(columns=row_data.keys())

    df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
    df.to_excel(file_path, sheet_name="resultados", index=False)
    logging.info(f"✅ Resultados salvos em {file_path}")


def display_class_distribution(type, dataset, emotion_table):
    class_counts = np.bincount(dataset.labels, minlength=len(emotion_table))
    logging.info(f"{type} class distribution:")    
    for idx, count in enumerate(class_counts):
        cname = emotion_table[idx]
        logging.info(f"  {cname:10s}: {count} ({count / len(dataset.labels) * 100:.2f}%)")
    
    logging.info(f"{type} dataset size: {len(dataset.labels)}\n")


def plot_confusion_matrix(cm, class_names, save_path=None):
    plot = ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        logging.info(f"✅ Matriz de confusão salva em: {save_path}")
        plt.close(plot.figure_)

    return plot.figure_

def display_sampler_distribution(train_loader, sampler_name=None, num_batches=50):
    """Mostra a distribuição de classes amostradas por um DataLoader."""
    sampled_labels = []
    for i, (_, y) in enumerate(train_loader):
        logged_labels = y.argmax(dim=1) if y.dim() > 1 else y
        sampled_labels.extend([int(label) for label in logged_labels])
        if i >= num_batches:
            break

    print("Distribuição amostrada (exemplo ~50 batches):")
    print(Counter(sampled_labels))
    if sampler_name is None and hasattr(train_loader.sampler, "__class__"):
        sampler_name = type(train_loader.sampler).__name__
    print(f"Sampler ativo: {sampler_name}")
