import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from math import floor
from random import shuffle


def get_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]
    )

class PlacesDataset(Dataset):
    def __init__(self,
                 dataset_npy: str,
                 mode: str = "train",
                 perc_train: float = 1.0,
                 eiil_output: bool = False):
        self.data = {}
        self.classes = {
            'bedroom': '0',
            'childs_room': '1',
        }
        self.idx_to_class = {str(idx): value for idx, value in enumerate(self.classes.keys())}
        self.mode = mode
        self.transform = get_transform()
        
        reader = np.load(dataset_npy)
        for [img_path, label] in reader:
            # label == classe name
            if not eiil_output:
                if label in self.classes.keys():
                    if self.classes[label] in self.data.keys(): 
                        self.data[self.classes[label]].append([img_path, self.classes[label]])
                    else:
                        self.data[self.classes[label]] = [[img_path, self.classes[label]]]
            else:
                # label == class idx
                if label in self.data.keys():
                    self.data[label].append([img_path, label])
                else:
                    self.data[label] = [[img_path, label]]

        if mode == "train":
            for label in self.data.keys():
                self.data[label] = self.data[label][:floor(0.8*len(self.data[label]))]
        if mode == "validation":
            for label in self.data.keys():
                self.data[label] = self.data[label][floor(0.8*len(self.data[label])):]
        
        print(self.data.keys())
        new_data = np.concatenate([
            np.asarray(self.data['0']),
            np.asarray(self.data['1'])
        ])
        self.data = new_data
        print(self.data.shape) # data = [['fullpath', 'label'], ....]
        
        labels, counts = np.unique(self.data[:, 1], return_counts = True)
        self.labels = labels.astype(int) # labels are integers folowing self.classes
        
        # Calculate class weights for WeightedRandomSampler
        self.class_counts = dict(zip(labels, counts))
        self.class_weights = {label: max(self.class_counts.values()) / count
                              for label, count in self.class_counts.items()}
        self.sampler_weights = [self.class_weights[cls] for cls in self.data[:, 1]]
        
        self.class_weights_list = [self.class_weights[k]
                                   for k in sorted(self.class_weights)]
        
        # Calculate class weights for CrossEntropyLoss (resnet50 train)
        self.class_simple_weights = [class_count/sum(counts) for class_count in counts]

        print('Found {} images from {} classes.'.format(len(self.data), len(self.labels)))
        for idx in self.class_counts.keys():
            print("    Class '{}' ({}): {} images.".format(
                  self.idx_to_class[idx], idx, self.class_counts[idx]))

    def __getitem__(self, index: int):
        img_path, label = self.data[index]
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        label = torch.tensor(int(label))
        return image, label, img_path

    def __len__(self):
        return len(self.data)

    def get_loader(self, train, batch_size, n_workers, sampler=None):
        if not train:
            shuffle = False
        else:
            shuffle = True
        loader = DataLoader(
            self,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=n_workers,
            sampler=sampler
        )
        return loader
