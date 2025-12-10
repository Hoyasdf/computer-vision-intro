import os
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class POC_Dataset(Dataset):
    def __init__(self, data_dir, data_type="Training", size=(224, 224), is_augment=False, transform=None, target_transform=None):
        self.size = size
        self.is_augment = is_augment
        
        self.data_dir = data_dir
        self.data_type = data_type
        
        self.image_names, self.labels = self.__process_data()
        
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        label = self.labels[idx]
        
        label_map = {0: 'Chorionic_villi', 1: 'Decidual_tissue', 2: 'Hemorrhage', 3: 'Trophoblastic_tissue'}
        
        image_path = os.path.join(self.data_dir, self.data_type, label_map[label], image_name)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)
            
        if self.is_augment:
            saved_image_path = os.path.join(self.data_dir, self.data_type, label_map[label], image_name.split('.')[0] +  '_augmented.jpg')
            while os.path.exists(saved_image_path):
                random_number = np.random.randint(0, 100)
                saved_image_path = saved_image_path.split('.')[0] + f'_{random_number}.jpg'

            image = transforms.RandomRotation(degrees=(0, 90))(image)
            image = transforms.RandomHorizontalFlip(p=1.0)(image)
            image.save(saved_image_path)
        
        return image, torch.tensor(label, dtype=torch.long)
                                  
    
    def __process_data(self):
        image_names = []
        labels = []
        
        cv = os.listdir(os.path.join(self.data_dir, self.data_type, 'Chorionic_villi'))
        dt = os.listdir(os.path.join(self.data_dir, self.data_type, 'Decidual_tissue'))
        h = os.listdir(os.path.join(self.data_dir, self.data_type, 'Hemorrhage'))
        tt = os.listdir(os.path.join(self.data_dir, self.data_type, 'Trophoblastic_tissue'))
        
        image_names = cv + dt + h + tt
        labels = [0]*len(cv) + [1]*len(dt) + [2]*len(h) + [3]*len(tt)
        
        return image_names, labels