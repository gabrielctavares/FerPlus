import sys
import os
import csv
import numpy as np
import logging
import random as rnd
from collections import namedtuple
from typing import List, Tuple, Optional, Union
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from rect_util import Rect

class FERPlusParameters:
    '''
    Parâmetros do leitor FER+ com melhor organização
    '''
    def __init__(self, target_size: int = 8, width: int = 48, height: int = 48, 
                 training_mode: str = "majority", deterministic: bool = False, 
                 shuffle: bool = True, max_shift: float = 0.08, max_scale: float = 1.05, 
                 max_angle: float = 20.0, max_skew: float = 0.05, do_flip: bool = True,
                 num_workers: int = 4, preload_data: bool = False):
        
        self.target_size = target_size
        self.width = width
        self.height = height
        self.training_mode = training_mode
        self.deterministic = deterministic
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.preload_data = preload_data

        # Parâmetros de aumento de dados
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

def load_image_data(args: Tuple[str, List[int], int, int], transform: Optional[transforms.Compose] = None) -> Tuple[torch.Tensor, List[int]]:
    '''
    Função para carregamento paralelo de imagens
    '''
    image_path, face_box, width, height = args
    try:
        # Carrega e processa a imagem
        image = Image.open(image_path).convert('L')
        
        # Aplica crop da face
        if face_box:
            left, top, right, bottom = face_box
            # Garante que as coordenadas estão dentro dos limites da imagem
            img_width, img_height = image.size
            left = max(0, min(left, img_width))
            right = max(left, min(right, img_width))
            top = max(0, min(top, img_height))
            bottom = max(top, min(bottom, img_height))
            
            if right > left and bottom > top:
                image = image.crop((left, top, right, bottom))
        
        # Redimensiona para o tamanho alvo
        image = image.resize((width, height), Image.LANCZOS)
        
        if not transform:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalização [-1, 1]
            ])
        

        image_tensor = transform(image)
        return image_tensor, face_box
        
    except Exception as e:
        logging.error(f"Erro ao carregar imagem {image_path}: {e}")
        # Retorna tensor vazio em caso de erro
        empty_tensor = torch.zeros((1, height, width))
        return empty_tensor, face_box

class FERPlusDataset(Dataset):
    '''
    Dataset PyTorch otimizado para FER+ com carregamento antecipado opcional
    '''
    def __init__(self, base_folder: str, sub_folders: List[str], label_file_name: str, 
                 parameters: FERPlusParameters, transform: Optional[transforms.Compose] = None):
        
        self.base_folder = base_folder
        self.sub_folders = sub_folders
        self.label_file_name = label_file_name
        self.emotion_count = parameters.target_size
        self.width = parameters.width
        self.height = parameters.height
        self.shuffle = parameters.shuffle
        self.training_mode = parameters.training_mode
        self.transform = transform
        self.preload_data = parameters.preload_data
        self.num_workers = parameters.num_workers
        
        # Inicializa o processador de emoções
        self.emotion_processor = EmotionProcessor(parameters.target_size)
        
        # Estruturas de dados
        self.data = []
        self.preloaded_images = []
        self.per_emotion_count = np.zeros(self.emotion_count, dtype=np.int64)
        
        # Carrega os dados
        self._load_folders()
        
        if self.shuffle:
            self._shuffle_data()
    
    def _load_folders(self):
        '''Carrega os metadados dos folders'''
        logging.info("Carregando metadados...")
        
        for folder_name in self.sub_folders:
            logging.info(f"Processando {os.path.join(self.base_folder, folder_name)}")
            folder_path = os.path.join(self.base_folder, folder_name)
            label_path = os.path.join(folder_path, self.label_file_name)
            
            with open(label_path, 'r') as csvfile:
                emotion_reader = csv.reader(csvfile)
                for row in emotion_reader:
                    image_path = os.path.join(folder_path, row[0])
                    
                    # Processa bounding box
                    face_box = list(map(int, row[1][1:-1].split(',')))
                    
                    # Processa emoções
                    emotion_raw = list(map(float, row[2:]))
                    emotion = self.emotion_processor.process_raw_emotion(emotion_raw, self.training_mode)
                    
                    # Verifica se a emoção principal é válida
                    idx_most_voted = np.argmax(emotion)
                    if idx_most_voted < (len(emotion) - 2):  # Não é unknown/non-face
                        # Remove unknown e non-face
                        emotion_final = emotion[:-2]
                        sum_emotion = sum(emotion_final)
                        
                        if sum_emotion > 0:
                            emotion_final = [e / sum_emotion for e in emotion_final]
                            self.data.append((image_path, emotion_final, face_box))
                            self.per_emotion_count[idx_most_voted] += 1
        
        logging.info(f"Total de imagens válidas: {len(self.data)}")
        
        # Carrega as imagens se solicitado
        if self.preload_data:
            self._preload_images()
    
    def _preload_images(self):
        '''Carrega todas as imagens na memória usando processamento paralelo'''
        logging.info("Carregando imagens na memória...")
        
        # Prepara argumentos para processamento paralelo
        load_args = [
            (img_path, face_box, self.width, self.height) 
            for img_path, _, face_box in self.data
        ]
        
        # Carrega imagens em paralelo
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(load_image_data, args, self.transform): i 
                      for i, args in enumerate(load_args)}
            
            # Inicializa lista de imagens pré-carregadas
            self.preloaded_images = [None] * len(self.data)
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    image_tensor, _ = future.result()
                    self.preloaded_images[idx] = image_tensor
                except Exception as e:
                    logging.error(f"Erro no carregamento da imagem {idx}: {e}")
                    # Cria tensor vazio em caso de erro
                    self.preloaded_images[idx] = torch.zeros((1, self.height, self.width))
        
        logging.info("Carregamento de imagens concluído!")
    
    def _shuffle_data(self):
        '''Embaralha os dados mantendo a correspondência'''
        indices = list(range(len(self.data)))
        rnd.shuffle(indices)
        
        # Reordena dados
        self.data = [self.data[i] for i in indices]
        
        # Reordena imagens pré-carregadas se existirem
        if self.preloaded_images:
            self.preloaded_images = [self.preloaded_images[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= len(self.data):
            raise IndexError(f"Índice {idx} fora do range [0, {len(self.data)})")
        
        image_path, emotion_labels, face_box = self.data[idx]
        
        # Obtém a imagem
        if self.preload_data and self.preloaded_images:
            image_tensor = self.preloaded_images[idx].clone()
        else:
            # Carregamento lazy
            image_tensor, _ = load_image_data((image_path, face_box, self.width, self.height), transform=self.transform)

        # Aplica transformações adicionais se fornecidas
        if self.transform:
            # Converte tensor para PIL para aplicar transforms
            to_pil = transforms.ToPILImage()
            image_pil = to_pil(image_tensor)
            image_tensor = self.transform(image_pil)
        
        # Processa target baseado no modo de treinamento
        target = self.emotion_processor.process_target(emotion_labels, self.training_mode)
        target_tensor = torch.from_numpy(target)
        
        return image_tensor, target_tensor

    def get_emotion_distribution(self) -> dict:
        '''Retorna a distribuição das emoções no dataset'''
        emotion_names = ['neutral', 'happiness', 'surprise', 'sadness', 
                        'anger', 'disgust', 'fear', 'contempt']
        
        distribution = {}
        total = self.per_emotion_count.sum()
        
        for i, name in enumerate(emotion_names):
            count = self.per_emotion_count[i]
            percentage = (count / total * 100) if total > 0 else 0
            distribution[name] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }
        
        return distribution    
    

class EmotionProcessor:
    def __init__(self, target_size: int = 8):
        self.target_size = target_size
        
    def process_raw_emotion(self, emotion_raw: List[float], mode: str) -> List[float]:
        '''
        Processa dados brutos de emoção baseado no modo de treinamento
        '''
        size = len(emotion_raw)
        emotion_unknown = [0.0] * size
        emotion_unknown[-2] = 1.0  # unknown emotion

        # Remove emoções com um único voto (remoção de outliers)
        emotion_raw = [0.0 if val < 1.0 + sys.float_info.epsilon else val for val in emotion_raw]
        sum_list = sum(emotion_raw)
        
        if sum_list == 0:
            return emotion_unknown
            
        if mode == 'majority':
            return self._process_majority(emotion_raw, sum_list, emotion_unknown)
        elif mode in ['probability', 'crossentropy']:
            return self._process_probability(emotion_raw, sum_list, emotion_unknown, size)
        elif mode == 'multi_target':
            return self._process_multi_target(emotion_raw, sum_list, emotion_unknown)
        else:
            raise ValueError(f"Modo de treinamento desconhecido: {mode}")

    def _process_majority(self, emotion_raw: List[float], sum_list: float, 
                         emotion_unknown: List[float]) -> List[float]:
        '''Processamento para modo majority'''
        maxval = max(emotion_raw)
        if maxval > 0.5 * sum_list:
            emotion = [0.0] * len(emotion_raw)
            emotion[np.argmax(emotion_raw)] = maxval
            return self._normalize_emotion(emotion)
        return emotion_unknown

    def _process_probability(self, emotion_raw: List[float], sum_list: float,
                           emotion_unknown: List[float], size: int) -> List[float]:
        '''Processamento para modo probability/crossentropy'''
        emotion = [0.0] * size
        temp_emotion_raw = emotion_raw.copy()
        sum_part = 0
        count = 0
        valid_emotion = True
        
        while sum_part < 0.75 * sum_list and count < 3 and valid_emotion:
            if not temp_emotion_raw or max(temp_emotion_raw) == 0:
                break
                
            maxval = max(temp_emotion_raw)
            for i in range(size):
                if temp_emotion_raw[i] == maxval:
                    emotion[i] = maxval
                    temp_emotion_raw[i] = 0
                    sum_part += emotion[i]
                    count += 1
                    
                    if i >= self.target_size:  # unknown ou non-face
                        valid_emotion = False
                        if sum(emotion) > maxval:
                            emotion[i] = 0
                            count -= 1
                        break
                        
        if sum(emotion) <= 0.5 * sum_list or count > 3:
            return emotion_unknown
            
        return self._normalize_emotion(emotion)

    def _process_multi_target(self, emotion_raw: List[float], sum_list: float,
                            emotion_unknown: List[float]) -> List[float]:
        '''Processamento para modo multi_target'''
        threshold = 0.3
        emotion = [val if val >= threshold * sum_list else 0.0 for val in emotion_raw]
        
        if sum(emotion) <= 0.5 * sum_list:
            return emotion_unknown
            
        return self._normalize_emotion(emotion)

    def _normalize_emotion(self, emotion: List[float]) -> List[float]:
        '''Normaliza as emoções para somar 1'''
        emotion_sum = sum(emotion)
        return [float(e) / emotion_sum for e in emotion] if emotion_sum > 0 else emotion

    def process_target(self, target: List[float], training_mode: str) -> np.ndarray:
        '''
        Processa o target baseado no modo de treinamento
        '''
        if training_mode in ['majority', 'crossentropy']:
            return np.array(target, dtype=np.float32)
        elif training_mode == 'probability':
            return self._process_probability_target(target)
        elif training_mode == 'multi_target':
            return self._process_multi_target_target(target)
        else:
            raise ValueError(f"Modo de treinamento desconhecido: {training_mode}")

    def _process_probability_target(self, target: List[float]) -> np.ndarray:
        '''Seleciona uma emoção baseada na distribuição de probabilidade'''
        idx = np.random.choice(len(target), p=target)
        new_target = np.zeros_like(target)
        new_target[idx] = 1.0
        return new_target.astype(np.float32)

    def _process_multi_target_target(self, target: List[float]) -> np.ndarray:
        '''Trata emoções com >0 como targets válidos com epsilon'''
        new_target = np.array(target)
        new_target[new_target > 0] = 1.0
        epsilon = 0.001
        return ((1 - epsilon) * new_target + epsilon * np.ones_like(target)).astype(np.float32)
