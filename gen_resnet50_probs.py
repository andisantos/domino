#!/usr/bin/env python
# coding: utf-8
import torchvision

from dataloader_places2 import PlacesDataset
from torchvision.models.feature_extraction import create_feature_extractor

# from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn.functional as nnf
import numpy as np


# idx_to_classname ={0: 'bathroom', 1: 'bedroom', 2: 'childs_room',
#                    3: 'classroom', 4: 'dressing_room', 5: 'living_room',
#                    6: 'studio', 7: "swimming_pool"}
idx_to_classname = {
    0: "bedroom",
    1: "childs_room"
}

data_path = "/home/andreza.santos/data/places8/places8_train.npy"
dataset = PlacesDataset(data_path)
batch_size = 1
n_classes = len(idx_to_classname.keys())
dataloader = dataset.get_loader(train=False, batch_size=batch_size, n_workers=6, sampler=None)
print(f"\nDataloader: {batch_size} batch size | {len(dataloader)} batches | {len(dataloader.dataset)} images")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model_path = "/home/andreza.santos/models/resnet50/pretraining/2c_SUN_MIT_bestmodel.pth"
model = torchvision.models.resnet50()
# model.fc = torch.nn.Linear(in_features=2048, out_features=n_classes, bias=True)
model.load_state_dict(torch.load(model_path))
model.eval()

# Create the feature extraction model
feature_extractor = create_feature_extractor(model, return_nodes={'avgpool': 'features'})

softmax_output_0 = [] 
softmax_output_1 = [] 
softmax_output_2 = [] 
softmax_output_3 = [] 
softmax_output_4 = [] 
softmax_output_5 = [] 
softmax_output_6 = []
softmax_output_7 = []
targets = []

for inputs, labels, _ in tqdm(dataloader):
    targets += list(labels.cpu().data.numpy())

    with torch.no_grad():
        features = feature_extractor(inputs)

    if labels.item() == 0:
        for out_prediction in features["features"]:
            features_list = out_prediction.squeeze().numpy().tolist()
            softmax_output_0.append(features_list)
    elif labels.item() == 1:
        for out_prediction in features["features"]:
            features_list = out_prediction.squeeze().numpy().tolist()
            softmax_output_1.append(features_list)

targets = np.asarray(targets)
print("targets shape", targets.shape)
np.save("data/places8_image_targets.npy", targets)


for idx, softmax_list in enumerate([softmax_output_0, softmax_output_1,
                                    softmax_output_2, softmax_output_3,
                                    softmax_output_4, softmax_output_5,
                                    softmax_output_6, softmax_output_7]):
    print(idx, len(softmax_list))
    if len(softmax_list) != 0:
        softmax_output = np.asarray(softmax_list)
        print(idx, softmax_output.shape)
        np.save(f"data/features_resnet50_class_{idx}.npy", softmax_output)
