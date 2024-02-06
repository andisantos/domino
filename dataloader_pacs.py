import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob


def get_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class Dataloader_PACS(Dataset):
    def __init__(self):
        self.data = []
        self.domains = {
            'art_painting': 0,
            'cartoon': 1,
            'photo': 2,
            'sketch': 3}
        source = '/hadatasets/andreza/datasets_oodbench/PACS'
        lista = glob(source+'/**/*.jpg', recursive=True)
        lista.extend(glob(source+'/**/*.png', recursive=True))
        for data_path in lista:
            splited = data_path.split('/')
            label = splited[-2]
            domain = splited[-3]
            self.data.append([data_path, label, domain])
            
        
        self.transform = get_transform()
        self.data = np.array(self.data) # data = [['fullpath', 'label', 'domain'], ....]
        labels = np.unique(self.data[:,1])
        self.class_to_idx = {
            value: int(idx) for idx, value in enumerate(labels)
        }

        
    def __getitem__(self, index: int):
        img_path, label, domain = self.data[index]
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        label = torch.tensor(int(self.class_to_idx[label]))
        return image, label, img_path, domain

    def __len__(self):
        return len(self.data)