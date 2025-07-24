import os
import csv
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging

# Dependências externas (certifique-se que estes módulos estejam disponíveis)
from rect_util import Rect
import img_util as imgu

class FERPlusDataset(Dataset):
    """
    Dataset para FER+ com modos `majority`, `probability`, `crossentropy` e `multi_target`.
    Refatorado para maior modularidade e robustez.

    Inclui método para plotar e registrar a distribuição de classes no TensorBoard.
    """
    def __init__(
        self,
        folder_paths,
        label_file_name,
        num_classes,
        width,
        height,
        mode='majority',
        shuffle=True,
        deterministic=False
    ):
        # Parâmetros básicos
        self.folder_paths = folder_paths
        self.label_file_name = label_file_name
        self.num_classes = num_classes  # ignora classes unknown e non-face
        self.width = width
        self.height = height
        self.mode = mode
        self.shuffle = shuffle

        # Parâmetros de augmentação
        if deterministic:
            self.aug_params = dict(max_shift=0.0, max_scale=1.0, max_angle=0.0, max_skew=0.0, flip=False)
        else:
            self.aug_params = dict(max_shift=0.08, max_scale=1.05, max_angle=20.0, max_skew=0.05, flip=True)

        # Pré-cálculo da matriz de normalização geométrica
        self.A, self.A_pinv = imgu.compute_norm_mat(width, height)

        # Carrega dados em memória
        self.data = self._load_data()
        if self.shuffle:
            np.random.shuffle(self.data)

    def _load_data(self):
        samples = []
        for folder in self.folder_paths:
            csv_path = os.path.join(folder, self.label_file_name)
            with open(csv_path, newline='') as f:
                reader = csv.reader(f)
                # next(reader)  # descomente se houver cabeçalho
                for row in reader:
                    parsed = self._parse_csv_row(folder, row)
                    if parsed is not None:
                        samples.append(parsed)
        return samples

    def _parse_csv_row(self, folder, row):
        # Monta caminho da imagem
        img_path = os.path.join(folder, row[0])
        if not os.path.isfile(img_path):
            logging.warning(f"Arquivo não encontrado: {img_path}")
            return None

        # Extrai box e cria Rect
        coords = row[1].strip('()').split(',')
        face_rc = Rect(list(map(int, coords)))

        # Votos brutos (float)
        raw_votes = list(map(float, row[2:]))

        # Processa distribuição de votos
        dist_full = self._process_data(raw_votes)
        if dist_full is None:
            return None

        # Descarta classes unknown e non-face
        valid_dist = dist_full[:self.num_classes]
        return (img_path, valid_dist, face_rc)

    def _process_data(self, raw_votes):
        # Copia e remove votos unitários
        votes = np.array(raw_votes, dtype=float)
        votes[votes <= 1.0 + sys.float_info.epsilon] = 0.0
        total = votes.sum()
        if total == 0:
            return None

        # Escolhe a estratégia
        if self.mode == 'majority':
            dist = self._pd_majority(votes)
        elif self.mode in ('probability', 'crossentropy'):
            dist = self._pd_probability(votes)
        elif self.mode == 'multi_target':
            dist = self._pd_multi_target(votes)
        else:
            raise ValueError(f"Modo inválido: {self.mode}")

        # Normaliza e remove as últimas 2 classes
        return self._normalize_and_strip(dist)

    def _pd_majority(self, votes):
        maxv = votes.max()
        total = votes.sum()
        if maxv > 0.5 * total:
            dist = np.zeros_like(votes)
            dist[np.argmax(votes)] = maxv
        else:
            dist = self._unknown_dist(votes.shape[0])
        return dist

    def _pd_probability(self, votes):
        # Lógica de probability/crossentropy original adaptada
        dist = np.zeros_like(votes)
        votes_copy = votes.copy()
        total = votes_copy.sum()
        sum_part = 0.0
        count = 0
        valid = True
        # adiciona top-k até acumular 75% ou até 3
        while sum_part < 0.75 * total and count < 3 and valid:
            maxv = votes_copy.max()
            idxs = np.where(votes_copy == maxv)[0]
            for i in idxs:
                if sum_part >= 0.75 * total or count >= 3:
                    break
                # descarta unknown/non-face
                if i >= self.num_classes:
                    valid = False
                    break
                dist[i] = maxv
                votes_copy[i] = 0.0
                sum_part += maxv
                count += 1
        if sum(dist) <= 0.5 * total or count == 0:
            dist = self._unknown_dist(votes.shape[0])
        return dist

    def _pd_multi_target(self, votes):
        threshold = 0.3 * votes.sum()
        dist = np.where(votes >= threshold, votes, 0.0)
        if dist.sum() <= 0.5 * votes.sum():
            dist = self._unknown_dist(votes.shape[0])
        return dist

    def _unknown_dist(self, size):
        d = np.zeros(size, dtype=float)
        d[-2] = 1.0
        return d

    def _normalize_and_strip(self, dist):
        total = dist.sum()
        if total > 0:
            dist = dist / total
        # remove unknown e non-face
        return dist[:-2]

    def _process_target(self, dist):
        arr = np.array(dist, dtype=np.float32)
        if self.mode in ('majority', 'crossentropy'):
            # one-hot
            one_hot = np.zeros_like(arr)
            one_hot[arr.argmax()] = 1.0
            return torch.from_numpy(one_hot)
        elif self.mode == 'probability':
            idx = np.random.choice(len(arr), p=arr)
            one_hot = np.zeros_like(arr)
            one_hot[idx] = 1.0
            return torch.from_numpy(one_hot)
        else:  # multi_target
            mask = np.where(arr > 0, 1.0, 0.0)
            eps = 1e-3
            soft = (1 - eps) * mask + eps
            return torch.from_numpy(soft.astype(np.float32))

    def plot_class_distribution(self, writer, step=0):
        """
        Plota a distribuição de classes (rótulo majoritário) e registra no TensorBoard.

        Args:
            writer (SummaryWriter): instância do TensorBoard writer.
            step (int): passo global para registro.
        """
        from collections import Counter
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image as PILImage

        # Extrai rótulo majoritário de cada exemplo
        labels = [dist.argmax() for _, dist, _ in self.data]
        counts = Counter(labels)
        classes = [f'c{i}' for i in range(self.num_classes)]
        values = [counts.get(i, 0) for i in range(self.num_classes)]

        # Gera gráfico
        plt.figure(figsize=(8, 4))
        plt.bar(classes, values)
        plt.xlabel('Classe')
        plt.ylabel('Contagem')
        plt.title('Distribuição de Classes')
        plt.tight_layout()

        # Salva em buffer e registra no TensorBoard
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = np.array(PILImage.open(buf))
        writer.add_image('Class Distribution', img, global_step=step, dataformats='HWC')
        plt.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, dist, face_rc = self.data[idx]
        # Abre imagem com robustez
        try:
            img = Image.open(img_path) 
            img.load()
        except Exception as e:
            logging.error(f"Erro ao abrir {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # Aplica distorção e pré-processamento
        img = imgu.distort_img(
            img, face_rc,
            self.width, self.height,
            **self.aug_params
        )
        proc = imgu.preproc_img(img, A=self.A, A_pinv=self.A_pinv)

        # Garante shape [C, H, W]
        arr = proc if proc.ndim == 3 else np.expand_dims(proc, 0)
        tensor = torch.from_numpy(arr).float()
        target = self._process_target(dist)
        return tensor, target
