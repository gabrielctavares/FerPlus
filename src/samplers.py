import itertools
import torch
from torch.utils.data import WeightedRandomSampler, RandomSampler, SequentialSampler, SubsetRandomSampler
from ferplus import FERPlusDataset
import logging

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

def get_sampler(sampler_type, dataset: FERPlusDataset, seed: int = 42, verbose: bool = False, **kwargs):
    if sampler_type is None or sampler_type.lower() == "none":
        return None
    if sampler_type not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown sampler type: {sampler_type}. Disponíveis: {list(SAMPLER_REGISTRY.keys())}")
    return SAMPLER_REGISTRY[sampler_type](dataset, seed=seed, verbose=verbose, **kwargs)


@register_sampler("weighted")
def weighted_sampler(dataset: FERPlusDataset, seed: int = 42, verbose: bool = False, **kwargs):
    class_counts = torch.tensor(dataset.per_emotion_count, dtype=torch.float)
    class_counts = torch.clamp(class_counts, min=1.0)

    # pesos proporcionais ao inverso da frequência (sem normalizar)
    class_weights = 1.0 / class_counts

    labels = dataset.labels if isinstance(dataset.labels, torch.Tensor) else torch.tensor(dataset.labels, dtype=torch.long)
    sample_weights = class_weights[labels]

    g = torch.Generator().manual_seed(seed)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=g
    )

    if verbose:
        logging.info("[weighted] Sampler configurado com balanceamento inverso.")
        _log_class_distribution(labels, sampler=sampler, name="weighted")

    return sampler

@register_sampler("balanced_per_class")
def balanced_per_class_sampler(dataset: FERPlusDataset, seed: int = 42, verbose: bool = False, **kwargs):
    labels = torch.tensor(dataset.labels, dtype=torch.long)
    class_counts = torch.tensor(dataset.per_emotion_count, dtype=torch.long)
    samples_per_class = class_counts[class_counts > 0].min().item()

    g = torch.Generator().manual_seed(seed)
    indices = []

    for c in range(len(class_counts)):
        class_indices = torch.where(labels == c)[0]
        if len(class_indices) == 0:
            logging.warning(f"[balanced_per_class] Classe {c} ignorada (sem amostras).")
            continue

        perm = torch.randperm(len(class_indices), generator=g)
        chosen = class_indices[perm[:samples_per_class]]
        indices.append(chosen)

    indices = torch.cat(indices)
    indices = indices[torch.randperm(len(indices), generator=g)]

    sampler = SubsetRandomSampler(indices)

    if verbose:
        logging.info(f"[balanced_per_class] {len(indices)} amostras (~ {samples_per_class} por classe).")
        _log_class_distribution(labels, indices, name="balanced_per_class")

    return sampler