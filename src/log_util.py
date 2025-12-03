import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns

def save_results_to_excel(file_path, row_data):
    import pandas as pd
    try:
        df = pd.read_excel(file_path, sheet_name="resultados")
    except FileNotFoundError:
        df = pd.DataFrame(columns=row_data.keys())

    df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
    df.to_excel(file_path, sheet_name="resultados", index=False)
    logging.info(f"✅ Resultados salvos em {file_path}")


def display_camadas_modelo(model):
    total = 0
    treinaveis = 0

    print(model)
    for name, param in model.named_parameters():
        total += 1
        if param.requires_grad:
            treinaveis += 1
            print(f"Treinável: {name}")
        else:
            print(f"Congelado: {name}")

    print(f"\nTotal de camadas: {total}")
    print(f"Camadas treináveis: {treinaveis}")
    print(f"Camadas congeladas: {total - treinaveis}")

def display_class_distribution(type, dataset, emotion_table):
    class_counts = np.bincount(dataset.labels, minlength=len(emotion_table))
    logging.info(f"{type} class distribution:")    
    for idx, count in enumerate(class_counts):
        cname = emotion_table[idx]
        logging.info(f"  {cname:10s}: {count} ({count / len(dataset.labels) * 100:.2f}%)")
    
    logging.info(f"{type} dataset size: {len(dataset.labels)}\n")


def plot_confusion_matrix(cm, class_names, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import logging

    # Normaliza a matriz
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Heatmap SEM colorbar
    sns.heatmap(
        cm_normalized,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax
    )

    # Inserindo manualmente os valores formatados
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            value = cm_normalized[i, j] * 100
            txt = f"{value:.2f}".replace(".", ",") + "%"
            ax.text(j + 0.5, i + 0.5, txt,
                    ha='center', va='center', color='black', fontsize=10)

    ax.set_title("Matriz de Confusão")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # SALVA SOMENTE EM PDF
    if save_path:
        if not save_path.lower().endswith(".pdf"):
            save_path += ".pdf"
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        logging.info(f"✅ Matriz de confusão salva em: {save_path}")

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
