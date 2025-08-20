#!/usr/bin/env python
# coding: utf-8
import torchvision

#from dataloader_places import PlacesDataset
from dataset_places8_2classes import PlacesDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn.functional as nnf
import numpy as np


idx_to_classname ={0: 'bathroom', 1: 'bedroom', 2: 'childs_room',
                   3: 'classroom', 4: 'dressing_room', 5: 'living_room',
                   6: 'studio', 7: "swimming_pool"}
idx_to_classname = {
    0: "bedroom",
    1: "childs_room"
}

data_path = "../adversarial-sets/data/Places8_paths_and_labels_complete_train.npy"
dataset = PlacesDataset(data_path, mode ="test")
batch_size = 1
n_classes = len(idx_to_classname.keys())
dataloader = DataLoader(dataset, shuffle=False, num_workers=6, batch_size=batch_size)
print(f"\nDataloader: {batch_size} batch size | {len(dataloader)} batches | {len(dataloader.dataset)} images")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model_path = "../default_training_nns/outputs/resnet50_SUN-MIT/2c_SUN_MIT_bestmodel.pth" #"resnet50_mit-sun/resnet50_mit-sun.pth"
model = torchvision.models.resnet50()
model.fc = torch.nn.Linear(in_features=2048, out_features=n_classes, bias=True)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()


softmax_output_0 = [] 
softmax_output_1 = [] 
softmax_output_2 = [] 
softmax_output_3 = [] 
softmax_output_4 = [] 
softmax_output_5 = [] 
softmax_output_6 = []

for inputs, labels, _ in tqdm(dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = torch.nn.Softmax(dim=1)(model(inputs))
        outputs = outputs.cpu().data.numpy()
        
        if labels == 0:
            for out_prediction in outputs:
                softmax_output_0.append(out_prediction.tolist())
        elif labels == 1:
            for out_prediction in outputs:
                softmax_output_1.append(out_prediction.tolist())
        elif labels == 2:
            for out_prediction in outputs:
                softmax_output_2.append(out_prediction.tolist())
        elif labels == 3:
            for out_prediction in outputs:
                softmax_output_3.append(out_prediction.tolist())
        elif labels == 4:
            for out_prediction in outputs:
                softmax_output_4.append(out_prediction.tolist())
        elif labels == 5:
            for out_prediction in outputs:
                softmax_output_5.append(out_prediction.tolist())
        elif labels == 6:
            for out_prediction in outputs:
                softmax_output_6.append(out_prediction.tolist())
        else:
            print('error')

for idx, softmax_list in enumerate([softmax_output_0, softmax_output_1,
                                    softmax_output_2, softmax_output_3,
                                    softmax_output_4, softmax_output_5,
                                    softmax_output_6]):
    print(idx, len(softmax_list))
    if len(softmax_list) != 0:
        softmax_output = np.asarray(softmax_list)
        np.save(f"data_sun-mit/features_resnet50_softmax_class_{idx}.npy", softmax_output)
