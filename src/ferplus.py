import sys
import os
import csv
import numpy as np
import logging
import random as rnd
from collections import namedtuple

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from rect_util import Rect

def display_summary(train_data_reader, val_data_reader, test_data_reader):
    '''
    Resume os dados em um formato tabular.
    '''
    emotion_count = train_data_reader.emotion_count
    emotin_header = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

    logging.info("{0}\t{1}\t{2}\t{3}".format("".ljust(10), "Train", "Val", "Test"))
    for index in range(emotion_count):
        logging.info("{0}\t{1}\t{2}\t{3}".format(emotin_header[index].ljust(10), 
                                                 train_data_reader.per_emotion_count[index], 
                                                 val_data_reader.per_emotion_count[index], 
                                                 test_data_reader.per_emotion_count[index]))

class FERPlusParameters():
    '''
    Parâmetros do leitor FER+
    '''
    def __init__(self, target_size, width, height, training_mode = "majority", deterministic = False, shuffle = True,
                 max_shift=0.08, max_scale=1.05, max_angle=20.0, max_skew=0.05, do_flip=True):
        self.target_size   = target_size
        self.width         = width
        self.height        = height
        self.training_mode = training_mode
        self.deterministic = deterministic # Corrigido para 'deterministic'
        self.shuffle       = shuffle

        # Parâmetros de aumento de dados (adicionados aqui)
        # Se 'deterministic' for True, sobrescreve com valores para sem aumento
        if self.deterministic:
            self.max_shift = 0.0
            self.max_scale = 1.0
            self.max_angle = 0.0
            self.max_skew = 0.0
            self.do_flip = False
        else:
            self.max_shift = max_shift
            self.max_scale = max_scale
            self.max_angle = max_angle
            self.max_skew = max_skew
            self.do_flip = do_flip
                        
class FERPlusDataset(Dataset):
    def __init__(self, base_folder, sub_folders, label_file_name,
                 parameters: FERPlusParameters, transform=None):
        self.base_folder     = base_folder
        self.sub_folders     = sub_folders
        self.label_file_name = label_file_name
        self.width           = parameters.width
        self.height          = parameters.height
        self.shuffle         = parameters.shuffle
        self.mode            = parameters.training_mode
        self.transform       = transform
        self.deterministic   = parameters.deterministic

        # pré-calcula matrizes de normalização (como no código antigo)
        self.A, self.A_pinv = imgu.compute_norm_mat(self.width, self.height)

        # parâmetros de distorção
        self.max_shift = parameters.max_shift
        self.max_scale = parameters.max_scale
        self.max_angle = parameters.max_angle
        self.max_skew  = parameters.max_skew
        self.do_flip   = parameters.do_flip

        self.emotion_count       = parameters.target_size
        self.data                = []
        self.per_emotion_count   = np.zeros(self.emotion_count, dtype=np.int64)
        self._load_folders()
        if self.shuffle:
            rnd.shuffle(self.data)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        image_path, label_dist, face_rc_box = self.data[idx]
        # abre e distorce com NumPy/C
        img = Image.open(image_path)
        img.load()
        aug = imgu.distort_img(
            img, Rect(face_rc_box),
            self.width, self.height,
            self.max_shift, self.max_scale,
            self.max_angle, self.max_skew, self.do_flip
        )
        proc = imgu.preproc_img(aug, A=self.A, A_pinv=self.A_pinv)
        tensor = torch.from_numpy(proc).float()

        # se houver transform TorchVision, aplica por cima
        if self.transform:
            tensor = self.transform(tensor)

        # target
        target = self._process_target(label_dist)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return tensor, target_tensor


    # def __getitem__(self, idx):
    #     image_path, emotion_labels, face_rc_box = self.data[idx] # face_rc_box é a lista [l,t,r,b]

    #     # Carrega a imagem do disco como PIL Image e a converte para escala de cinza
    #     image = Image.open(image_path).convert('L') 
        
    #     # OFERECER DUAS ABORDAGENS para o recorte da face:
    #     # ABORDAGEM 1: Recorte manual (mais próximo do original se Rect fosse crucial para o recorte)
    #     # left, top, right, bottom = face_rc_box
    #     # cropped_image = image.crop((left, top, right, bottom))
    #     # image_to_transform = cropped_image
        
    #     # ABORDAGEM 2: Se a bounding box é apenas para informação, e o dataset já tem faces centralizadas/recortadas,
    #     # ou se queremos que o modelo aprenda a ignorar o fundo (menos provável para FER+).
    #     # Para FER+, o dataset já é de faces recortadas, então basta redimensionar.
    #     image_to_transform = image

    #     # Aplica as transformações do TorchVision
    #     if self.transform:
    #         image_tensor = self.transform(image_to_transform)
    #     else:
    #         # Se nenhuma transformação for fornecida, ainda precisamos de ToTensor e normalização básica
    #         # para converter a imagem PIL para um tensor PyTorch.
    #         image_tensor = transforms.Compose([
    #             transforms.Resize((self.height, self.width)), # Redimensiona para o tamanho alvo
    #             transforms.ToTensor(), # Converte PIL Image para FloatTensor e normaliza [0.0, 1.0]
    #         ])(image_to_transform)

    #     # Processa os rótulos de destino
    #     target = self._process_target(emotion_labels)
    #     target_tensor = torch.tensor(target, dtype=torch.float32)

    #     return image_tensor, target_tensor
            
    def _load_folders(self):
        for folder_name in self.sub_folders: 
            logging.info(f"Carregando {os.path.join(self.base_folder, folder_name)}")
            folder_path = os.path.join(self.base_folder, folder_name)
            in_label_path = os.path.join(folder_path, self.label_file_name)
            with open(in_label_path) as csvfile: 
                emotion_label = csv.reader(csvfile) 
                # Pula o cabeçalho do CSV se houver um
                # next(emotion_label, None) 
                for row in emotion_label: 
                    image_path = os.path.join(folder_path, row[0])
                    
                    # A bounding box será uma lista de ints, não um objeto Rect aqui.
                    face_rc_box = list(map(int, row[1][1:-1].split(',')))

                    emotion_raw = list(map(float, row[2:len(row)]))
                    emotion = self._process_data(emotion_raw, self.training_mode) 
                    
                    # A indexação de emoções deve ser cuidadosamente verificada.
                    # As emoções originais do FER+ são:
                    # 0: neutral, 1: happiness, 2: surprise, 3: sadness, 4: anger, 5: disgust, 6: fear, 7: contempt
                    # 8: unknown, 9: non-face
                    # Se target_size=8, estamos focando nas 8 primeiras.
                    # As emoções 'unknown' e 'non-face' são tratadas no _process_data
                    # e removidas antes de adicionar ao self.data se a emoção principal
                    # não for uma das 8 básicas.
                    
                    # Encontra o índice da emoção com maior voto APÓS o _process_data (que já filtra)
                    idx_most_voted = np.argmax(emotion)

                    # Se a emoção mais votada não for 'unknown' ou 'non-face' (últimos dois índices)
                    if idx_most_voted < (len(emotion) - 2): 
                        # Remove 'unknown' e 'non-face' das emoções finais a serem armazenadas
                        emotion_final = emotion[:-2] 
                        # Normaliza as probabilidades para que somem 1
                        sum_emotion_final = sum(emotion_final)
                        if sum_emotion_final > 0:
                            emotion_final = [float(i)/sum_emotion_final for i in emotion_final]
                        else:
                            emotion_final = [0.0] * self.emotion_count # Se a soma for 0, todas são 0
                        
                        self.data.append((image_path, emotion_final, face_rc_box))
                        # Conta apenas as 8 emoções principais
                        self.per_emotion_count[idx_most_voted] += 1
            
    def _process_target(self, target):
        # Esta função permanece a mesma, pois lida com a lógica dos rótulos
        if self.training_mode == 'majority' or self.training_mode == 'crossentropy': 
            return target
        elif self.training_mode == 'probability': 
            idx       = np.random.choice(len(target), p=target) 
            new_target      = np.zeros_like(target)
            new_target[idx] = 1.0
            return new_target
        elif self.training_mode == 'multi_target': 
            new_target = np.array(target) 
            new_target[new_target>0] = 1.0
            epsilon = 0.001       
            return (1-epsilon)*new_target + epsilon*np.ones_like(target)

    def _process_data(self, emotion_raw, mode):
        # Esta função permanece a mesma em sua lógica, pois decide como interpretar os votos brutos
        size = len(emotion_raw) # Inclui 'unknown' e 'non-face' aqui
        emotion_unknown       = [0.0] * size
        emotion_unknown[-2] = 1.0 

        for i in range(size):
            if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
                emotion_raw[i] = 0.0

        sum_list = sum(emotion_raw)
        emotion = [0.0] * size 

        if mode == 'majority': 
            maxval = max(emotion_raw) 
            if maxval > 0.5*sum_list: 
                emotion[np.argmax(emotion_raw)] = maxval 
            else: 
                emotion = emotion_unknown   
        elif (mode == 'probability') or (mode == 'crossentropy'):
            sum_part = 0
            count = 0
            valid_emotion = True
            temp_emotion_raw = list(emotion_raw) 
            while sum_part < 0.75*sum_list and count < 3 and valid_emotion:
                maxval = max(temp_emotion_raw) 
                if maxval == 0: 
                    break
                for i in range(size): 
                    if temp_emotion_raw[i] == maxval: 
                        emotion[i] = maxval
                        temp_emotion_raw[i] = 0 
                        sum_part += emotion[i]
                        count += 1
                        if i >= (self.emotion_count): # Se a emoção mais votada é unknown ou non-face (aqui self.emotion_count já é 8)
                            valid_emotion = False
                            if sum(emotion) > maxval:    
                                emotion[i] = 0
                                count -= 1
                            break
            if sum(emotion) <= 0.5*sum_list or count > 3: 
                emotion = emotion_unknown   
        elif mode == 'multi_target':
            threshold = 0.3
            for i in range(size): 
                if emotion_raw[i] >= threshold*sum_list: 
                    emotion[i] = emotion_raw[i] 
            if sum(emotion) <= 0.5 * sum_list: 
                emotion = emotion_unknown   
                                        
        current_sum = sum(emotion)
        if current_sum > 0:
            return [float(i)/current_sum for i in emotion]
        else:
            return [0.0] * size

