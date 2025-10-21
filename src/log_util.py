import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from sklearn.metrics import confusion_matrix


def save_results_to_excel(file_path, row_data):
    """Salva ou atualiza os resultados em um arquivo Excel."""
    sheet_name = "resultados"

    if os.path.exists(file_path):
        try:
            df_existente = pd.read_excel(file_path, sheet_name=sheet_name)
        except ValueError:
            df_existente = pd.DataFrame(columns=row_data.keys())
    else:
        df_existente = pd.DataFrame(columns=row_data.keys())

    df_final = pd.concat([df_existente, pd.DataFrame([row_data])], ignore_index=True)
    mode = 'a' if os.path.exists(file_path) else 'w'
    with pd.ExcelWriter(file_path, engine='openpyxl', mode=mode, if_sheet_exists='replace') as writer:
        df_final.to_excel(writer, sheet_name=sheet_name, index=False)
    logging.info(f"✅ Resultados salvos em {file_path}")


def display_class_distribution(type, dataset, emotion_table):
    class_counts = np.bincount(dataset.labels, minlength=len(emotion_table))
    logging.info(f"{type} class distribution:")    
    for idx, count in enumerate(class_counts):
        cname = emotion_table[idx]
        logging.info(f"  {cname:10s}: {count} ({count / len(dataset.labels) * 100:.2f}%)")
    
    logging.info(f"{type} dataset size: {len(dataset.labels)}\n")


def plot_confusion_matrix(labels, preds, class_names, save_path=None):
    """Gera e retorna um gráfico Matplotlib da matriz de confusão."""
    cm = confusion_matrix(labels, preds)
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    thresh = cm_norm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm_norm[i, j] > thresh else "black"
        plt.text(j, i, f"{cm_norm[i, j]:.2f}", horizontalalignment="center", color=color)

    plt.ylabel("Real")
    plt.xlabel("Predito")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logging.info(f"✅ Matriz de confusão salva em: {save_path}")
        plt.close(fig)

    return fig


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
