
import os
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from rect_util import Rect
import img_util as imgu


import cv2



class FERPlusParameters:
    def __init__(self, target_size, width, height, training_mode="majority", determinisitc=False, shuffle=True):
        self.target_size = target_size
        self.width = width
        self.height = height
        self.training_mode = training_mode
        self.determinisitc = determinisitc
        self.shuffle = shuffle

class FERPlusDataset(Dataset):
    def __init__(self, base_folder, sub_folders, label_file_name, parameters: FERPlusParameters):
        self.base_folder = base_folder
        self.sub_folders = sub_folders
        self.label_file_name = label_file_name
        self.emotion_count = parameters.target_size
        self.width = parameters.width
        self.height = parameters.height
        self.shuffle = parameters.shuffle
        self.training_mode = parameters.training_mode

        # data augmentation parameters
        if parameters.determinisitc:
            self.max_shift = 0.0
            self.max_scale = 1.0
            self.max_angle = 0.0
            self.max_skew = 0.0
            self.do_flip = False
        else:
            self.max_shift = 0.08
            self.max_scale = 1.05
            self.max_angle = 20.0
            self.max_skew = 0.05
            self.do_flip = True

        self.data = []
        self.labels = []
        self.per_emotion_count = np.zeros(self.emotion_count, dtype=np.int64)

        self.A, self.A_pinv = imgu.compute_norm_mat(self.width, self.height)
        self._load_folders(self.training_mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #image_path, image_data, emotion, face_rc = self.data[idx]
        image_path, emotion, face_rc = self.data[idx]
        
        # carregamento lazy para melhorar o desempenho no colab
        image_data = Image.open(image_path)
        image_data.load()  

        distorted = imgu.distort_img(
            np.asarray(image_data, dtype=np.uint8),
            face_rc,
            self.width,
            self.height,
            self.max_shift,
            self.max_scale,
            self.max_angle,
            self.max_skew,
            flip=self.do_flip
        )
        final_image = imgu.preproc_img(distorted, A=self.A, A_pinv=self.A_pinv) 

        #show final_image
        cv2.imshow("final_image", final_image)
        cv2.waitKey(0)
        # Convert to torch tensors in (C,H,W) with C=1
        
        x = torch.from_numpy(final_image.astype(np.float32)).unsqueeze(0)        
        y = torch.from_numpy(self._process_target(emotion).astype(np.float32))
        return x, y

    def _load_folders(self, mode):
        self.data.clear()
        self.per_emotion_count[:] = 0

        for folder_name in self.sub_folders:
            folder_path = os.path.join(self.base_folder, folder_name)
            in_label_path = os.path.join(folder_path, self.label_file_name)
            with open(in_label_path, newline='') as csvfile:
                emotion_label = csv.reader(csvfile)
                for row in emotion_label:
                    image_path = os.path.join(folder_path, row[0])
                    #image_data = Image.open(image_path) carregar em lazy pra melhorar o desempenho no colab.
                    #image_data.load()

                    box = list(map(int, row[1][1:-1].split(',')))
                    face_rc = Rect(box)

                    emotion_raw = list(map(float, row[2:len(row)]))
                    emotion = self._process_data(emotion_raw, mode)
                    idx = int(np.argmax(emotion))
                    if idx < self.emotion_count:  # not unknown or non-face
                        emotion = emotion[:-2]
                        s = float(sum(emotion))
                        emotion = [float(i)/s for i in emotion]

                        #self.data.append((image_path, image_data, np.array(emotion, dtype=np.float32), face_rc))
                        self.data.append((image_path, np.array(emotion, dtype=np.float32), face_rc))
                        self.labels.append(idx) 
                        self.per_emotion_count[idx] += 1

        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)
            self.data = [self.data[i] for i in self.indices]

    def _process_target(self, target):
        if self.training_mode in ('majority', 'crossentropy'):
            return np.asarray(target, dtype=np.float32)
        elif self.training_mode == 'probability':
            idx = np.random.choice(len(target), p=target)
            new_target = np.zeros_like(target, dtype=np.float32)
            new_target[idx] = 1.0
            return new_target
        elif self.training_mode == 'multi_target':
            new_target = np.array(target, dtype=np.float32)
            new_target[new_target > 0] = 1.0
            epsilon = 0.001
            return (1 - epsilon) * new_target + epsilon * np.ones_like(target, dtype=np.float32)
        else:
            raise ValueError(f"Unknown training_mode: {self.training_mode}")

    def _process_data(self, emotion_raw, mode):
        size = len(emotion_raw)
        emotion_unknown = [0.0] * size
        emotion_unknown[-2] = 1.0

        # outlier removal: remove single votes
        for i in range(size):
            if emotion_raw[i] < 1.0 + np.finfo(float).eps:
                emotion_raw[i] = 0.0

        sum_list = sum(emotion_raw)
        emotion = [0.0] * size

        if mode == 'majority':
            maxval = max(emotion_raw)
            if maxval > 0.5 * sum_list:
                emotion[int(np.argmax(emotion_raw))] = maxval
            else:
                emotion = emotion_unknown
        elif mode in ('probability', 'crossentropy'):
            sum_part = 0.0
            count = 0
            valid_emotion = True
            emo_work = list(emotion_raw)
            while sum_part < 0.75 * sum_list and count < 3 and valid_emotion:
                maxval = max(emo_work)
                for i in range(size):
                    if emo_work[i] == maxval:
                        emotion[i] = maxval
                        emo_work[i] = 0
                        sum_part += emotion[i]
                        count += 1
                        if i >= 8:  # unknown or non-face share max votes
                            valid_emotion = False
                            if sum(emotion) > maxval:  # there were other emotions already
                                emotion[i] = 0
                                count -= 1
                            break
            if sum(emotion) <= 0.5 * sum_list or count > 3:
                emotion = emotion_unknown
        elif mode == 'multi_target':
            threshold = 0.3
            for i in range(size):
                if emotion_raw[i] >= threshold * sum_list:
                    emotion[i] = emotion_raw[i]
            if sum(emotion) <= 0.5 * sum_list:
                emotion = emotion_unknown
        else:
            raise ValueError(f"Unknown mode: {mode}")

        s = float(sum(emotion)) if sum(emotion) != 0 else 1.0
        return [float(i) / s for i in emotion]
