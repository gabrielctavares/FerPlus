import itertools
import torch
from torch.utils.data import WeightedRandomSampler, RandomSampler, SequentialSampler, SubsetRandomSampler
from ferplus import FERPlusDataset
import logging
import numpy as np

SAMPLER_REGISTRY = {}

def _log_class_distribution(labels, sampler=None, name="sampler", sample_size=5000):
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, device="cuda" if torch.cuda.is_available() else "cpu")

    if sampler is not None:
        # pega até sample_size índices do sampler
        indices = torch.tensor(list(sampler)[:sample_size], device=labels.device)
        labels = labels[indices]

    unique, counts = torch.unique(labels, return_counts=True)
    dist = {int(k.item()): int(v.item()) for k, v in zip(unique, counts)}

    logging.info(f"[{name}] Distribuição amostral (estimada) por classe: {dist}")
    return dist
    
def register_sampler(name):
    def decorator(fn):
        SAMPLER_REGISTRY[name] = fn
        return fn
    return decorator

def get_sampler(sampler_type, dataset, seed: int = 42, verbose: bool = False, **kwargs):
    if sampler_type is None or sampler_type.lower() == "none":
        return None
    if sampler_type not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown sampler type: {sampler_type}. Disponíveis: {list(SAMPLER_REGISTRY.keys())}")
    return SAMPLER_REGISTRY[sampler_type](dataset, seed=seed, verbose=verbose, **kwargs)


@register_sampler("weighted")
def weighted_sampler(
    dataset: FERPlusDataset, 
    seed: int = 42, 
    verbose: bool = False, 
    epsilon: float = 1e-6,
    **kwargs
):
   
    labels_list = [t[1] for t in dataset.data]
    labels_tensor = torch.from_numpy(np.array(labels_list, dtype=np.float32))
  
    class_counts = torch.tensor(dataset.per_emotion_count, dtype=torch.float32)
    class_weights = 1.0 / (class_counts + epsilon)
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    if verbose:
        print(f"Shape do labels_tensor: {labels_tensor.shape}")
        print(f"Pesos das classes: {class_weights}")

    sample_weights = (labels_tensor * class_weights).sum(dim=1)
    
    if torch.any(sample_weights <= 0):
        if verbose:
            print("Aviso: Encontrados pesos não-positivos. Aplicando correção.")
        sample_weights = torch.clamp(sample_weights, min=epsilon)
    
    # Normaliza os pesos para soma = 1 (opcional, mas recomendado)
    #sample_weights = sample_weights / sample_weights.sum()

    if verbose:
        print(f"Estatísticas dos sample_weights:")
        print(f"  - Min: {sample_weights.min():.6f}")
        print(f"  - Max: {sample_weights.max():.6f}")
        print(f"  - Mean: {sample_weights.mean():.6f}")
        print(f"  - Std: {sample_weights.std():.6f}")
        print(f"Exemplos de sample_weights: {sample_weights[:6].tolist()}")

    generator = torch.Generator().manual_seed(seed)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels_list),
        replacement=True,
        generator=generator
    )


@register_sampler("weighted_soft")
def weighted_soft_sampler(
    dataset: FERPlusDataset, 
    seed: int = 42, 
    verbose: bool = False, 
    epsilon: float = 1e-6,
    **kwargs
):
    """
     Semelhante ao weighted_sampler, mas o peso das classes é calculado com base na soma das contribuições de cada classe em todas as amostras.
    """
   
    labels_list = [t[1] for t in dataset.data]
    labels_tensor = torch.from_numpy(np.array(labels_list, dtype=np.float32))

    if verbose:
        print(f"Total de amostras no dataset: {len(labels_list)}")
        print(f"Shape dos labels: {labels_tensor.shape}")
        print(f"Exemplo de label: {labels_tensor[0]}")
        print(f"Soma do exemplo: {labels_tensor[0].sum():.4f}")  # Deve ser ~1.0

    total_class_contributions = labels_tensor.sum(dim=0)
    
    if verbose:
        print(f"Contribuições totais por classe: {total_class_contributions}")

    class_weights = 1.0 / (total_class_contributions + epsilon)
    
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    if verbose:
        print(f"Pesos das classes: {class_weights}")

    sample_weights = (labels_tensor * class_weights).sum(dim=1)    
    sample_weights = torch.clamp(sample_weights, min=epsilon)

    

    if verbose:
        print(f"Estatísticas dos sample_weights:")
        print(f"  - Min: {sample_weights.min():.6f}")
        print(f"  - Max: {sample_weights.max():.6f}") 
        print(f"  - Mean: {sample_weights.mean():.6f}")
        print(f"  - Std: {sample_weights.std():.6f}")
        
        print("\nExemplos de amostras e seus pesos:")
        for i in range(min(5, len(sample_weights))):
            print(f"  Amostra {i}: label={labels_tensor[i]}, peso={sample_weights[i]:.6f}")

    generator = torch.Generator().manual_seed(seed)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels_list),
        replacement=True,
        generator=generator
    )

    return sampler


@register_sampler("balanced_per_class")
def weighted_balanced_sampler(dataset: FERPlusDataset, seed: int = 42, verbose: bool = False, **kwargs):
    labels_list = [t[1] for t in dataset.data]
    labels_tensor = torch.from_numpy(np.array(labels_list, dtype=np.float32))
    dominant_classes = torch.argmax(labels_tensor, dim=1)
    
    # Peso igual para todas as classes
    class_weights = torch.ones(len(dataset.per_emotion_count), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()
    
    sample_weights = class_weights[dominant_classes]
    
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels_list),
        replacement=True,
        generator=generator
    )

@register_sampler("affectnet_weighted")
def affectnet_weighted_sampler(dataset, seed: int = 42, verbose: bool = False, **kwargs):
    """
    Sampler ponderado específico para o dataset AffectNet, baseado na distribuição de classes do dataset original.
    """    

    targets = [label for _, label in dataset.samples]

    class_count = torch.bincount(torch.tensor(targets))
    class_weights = 1.0 / class_count.float()

    sample_weights = [class_weights[t] for t in targets]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,  
        generator=torch.Generator().manual_seed(seed)      
    )
    if verbose:
        _log_class_distribution(targets, sampler=sampler, name="AffectNet Weighted Sampler")

    return sampler
