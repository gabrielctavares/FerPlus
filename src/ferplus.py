import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from rect_util import Rect
import img_util as imgu  # adapta conforme original

import sys

class FERPlusDataset(Dataset):
    def __init__(self, folder_paths, label_file_name, num_classes, width, height,
                 mode='majority', shuffle=True, deterministic=False):
        self.folder_paths = folder_paths
        self.label_file_name = label_file_name
        self.num_classes = num_classes  # classes válidas (exclui unknown e non-face)
        self.width = width
        self.height = height
        self.mode = mode
        self.shuffle = shuffle

        if deterministic:
            self.max_shift = 0.0; self.max_scale = 1.0
            self.max_angle = 0.0; self.max_skew = 0.0; self.do_flip = False
        else:
            self.max_shift = 0.08; self.max_scale = 1.05
            self.max_angle = 20.0; self.max_skew = 0.05; self.do_flip = True

        self.A, self.A_pinv = imgu.compute_norm_mat(self.width, self.height)
        self.data = []
        self._load_data()

    def _load_data(self):
        for folder in self.folder_paths:
            path = os.path.join(folder, self.label_file_name)
            with open(path, newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    img_path = os.path.join(folder, row[0])
                    box = list(map(int, row[1][1:-1].split(',')))
                    face_rc = Rect(box)
                    raw = list(map(float, row[2:]))

                    dist = self._process_data(raw)
                    if dist is not None:
                        # descarta unknown e non-face (últimos 2)
                        valid_dist = dist[:self.num_classes]
                        self.data.append((img_path, valid_dist, face_rc))
        if self.shuffle:
            np.random.shuffle(self.data)

    # Substitua todo o método _process_data por:
    def _process_data(self, emotion_raw):
        size = len(emotion_raw)
        emotion_unknown = [0.0] * size
        emotion_unknown[-2] = 1.0  # unknown

        # Remove votos únicos
        for i in range(size):
            if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
                emotion_raw[i] = 0.0

        sum_list = sum(emotion_raw)
        emotion = [0.0] * size
        
        if self.mode == 'majority':
            maxval = max(emotion_raw)
            if maxval > 0.5 * sum_list:
                emotion[np.argmax(emotion_raw)] = maxval
            else:
                emotion = emotion_unknown
        elif self.mode in ('probability', 'crossentropy'):
            sum_part = 0
            count = 0
            valid_emotion = True
            while sum_part < 0.75 * sum_list and count < 3 and valid_emotion:
                maxval = max(emotion_raw)
                for i in range(size):
                    if emotion_raw[i] == maxval:
                        emotion[i] = maxval
                        emotion_raw[i] = 0
                        sum_part += emotion[i]
                        count += 1
                        if i >= 8:  # unknown ou non-face
                            valid_emotion = False
                            if sum(emotion) > maxval:
                                emotion[i] = 0
                                count -= 1
                            break
            if sum(emotion) <= 0.5 * sum_list or count > 3:
                emotion = emotion_unknown
        elif self.mode == 'multi_target':
            threshold = 0.3
            for i in range(size):
                if emotion_raw[i] >= threshold * sum_list:
                    emotion[i] = emotion_raw[i]
            if sum(emotion) <= 0.5 * sum_list:
                emotion = emotion_unknown

        # Filtra classes desconhecidas
        emotion = emotion[:-2]
        return [float(i)/sum(emotion) for i in emotion] if sum(emotion) > 0 else emotion


    def _process_target(self, dist):
        dist = np.array(dist, dtype=np.float32)

        if self.mode in ('majority', 'crossentropy'):
            return dist

        elif self.mode == 'probability':
            valid_indices = np.where(dist > 0)[0]

            if len(valid_indices) == 0:
                # fallback: escolhe uniformemente entre todas as classes
                idx = np.random.choice(len(dist))
            else:
                filtered = dist[valid_indices]
                p = filtered / np.sum(filtered)
                idx = np.random.choice(valid_indices, p=p)

            target = np.zeros(len(dist), dtype=np.float32)
            target[idx] = 1.0
            return target

        else:  # multi_target
            arr = dist.copy()
            arr[arr > 0] = 1.0
            eps = 0.001
            return (1 - eps) * arr + eps * np.ones_like(arr)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img_path, dist, face_rc = self.data[idx]
        img = Image.open(img_path); img.load()
        aug = imgu.distort_img(img, face_rc,
            self.width, self.height,
            self.max_shift, self.max_scale,
            self.max_angle, self.max_skew, self.do_flip)
        proc = imgu.preproc_img(aug, A=self.A, A_pinv=self.A_pinv)
        # garante shape [C,H,W]
        arr=proc if proc.ndim==3 else np.expand_dims(proc,0)
        tensor = torch.from_numpy(arr).float()
        target = torch.from_numpy(self._process_target(dist)).long() if self.mode!='multi_target' else torch.from_numpy(self._process_target(dist)).float()
        return tensor, target
