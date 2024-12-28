import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from math import floor
from random import shuffle


def get_transform(input_size=64):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class PlacesDataset(Dataset):
    def __init__(self,
                 dataset_npy: str,
                 mode: str = 'train',
                 perc_train: float = 1.0,
                 onlylabels = None,
                 eiil_output: bool = False
                ):
        self.data = []
        self.classes = {
            'bathroom': 0,
            'bedroom': 1,
            'childs_room': 2,
            'classroom': 3,
            'dressing_room': 4,
            'living_room': 5,
            'studio': 6,
            'swimming_pool': 7
        }
        self.idx_to_class = {str(idx): value for idx, value in enumerate(self.classes.keys())}
        self.mode = mode
        self.onlylabels = onlylabels
        self.transform = get_transform()
        
        reader = np.load(dataset_npy)
        for [img_path, label] in reader:
            if not eiil_output:
                # label == classe name
                self.data.append((img_path, self.classes[label])) # for some reason, label is str
            else:
                # label == class idx
                self.data.append((img_path, label))
        self.data = np.array(self.data) # data = [['fullpath', 'label'], ....]
#         shuffle(self.data)
        
        if mode == 'train':
            self.data = self.data[:floor(perc_train*len(self.data))]
        if mode == 'val':
            self.data = self.data[floor(perc_train*len(self.data)):]
            
        if self.onlylabels is not None:
            clip_indexes = np.where(self.data[:, 1] == str(self.onlylabels[0]))[0] # indexes
            for i in self.onlylabels[1:]:
                clip = np.where(self.data[:, 1] == str(i))[0]
                clip_indexes = np.append(clip_indexes, clip)
            clip_indexes.sort()
            self.data = self.data[clip_indexes]
        labels, counts = np.unique(self.data[:, 1], return_counts = True)
        self.labels = labels.astype(int) # labels are integers folowing self.classes
        
        # Calculate class weights for WeightedRandomSampler
        self.class_counts = dict(zip(labels, counts))
        self.class_weights = {label: max(self.class_counts.values()) / count
                              for label, count in self.class_counts.items()}
        self.sampler_weights = [self.class_weights[cls] for cls in self.data[:, 1]]
        
        self.class_weights_list = [self.class_weights[k]
                                   for k in sorted(self.class_weights)]

        print('Found {} images from {} classes.'.format(len(self.data),
                                                        len(self.labels)))
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

    def get_loader(self, train, batch_size, n_workers):
        if not train:
            shuffle = False
        else:
            shuffle = True
        loader = DataLoader(
            self,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=n_workers
        )
        return loader
