#!/usr/bin/env python
# coding: utf-8


from dataloader_places import PlacesDataset
from torch.utils.data import DataLoader
import numpy as np
import clip
import torch
from tqdm import tqdm

idx_to_classname ={0: 'bathroom', 1: 'bedroom', 2: 'childs_room',
                   3: 'classroom', 4: 'dressing_room', 5: 'living_room',
                   6: 'studio', 7: "swimming_pool"}

data_path = "../adversarial-sets/data/Places8_paths_and_labels_complete_train.npy"
places_ds = [PlacesDataset(data_path,
                           onlylabels=[k]) for k in range(8)]
batch_size = 64
train_dataloaders_class = {k: DataLoader(places_ds[k],
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=8) for k in range(8)}
for k in range(8):
    print(f"\nDataloader: {batch_size} batch size | {len(train_dataloaders_class[k])} batches | {len(train_dataloaders_class[k].dataset)} images")


print(clip.available_models())
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model, preprocess = clip.load("ViT-B/32", device=device)
print(model)

print(preprocess)

for i in range(8):
    image_features = []
    text_features = []
    for inputs, labels, _ in tqdm(train_dataloaders_class[i]):
        inputs = inputs.to(device)
        text_inputs = torch.cat([clip.tokenize(
            f"a photo of a {idx_to_classname[label.item()]}", truncate=True) for label in labels]).to(device)

        # Calculate features
        with torch.no_grad():
            img_feat_batch = model.encode_image(inputs).cpu().data.numpy()
            text_feat_batch = model.encode_text(text_inputs).cpu().data.numpy()
            for idx, _ in enumerate(img_feat_batch):
                image_features.append(img_feat_batch[idx].tolist())
                text_features.append(text_feat_batch[idx].tolist())
    
    image_features = np.asarray(image_features)
    text_features = np.asarray(text_features)
    np.save(f"places8_image_features_clip_class_{i}.npy", image_features)
    np.save(f"places8_text_features_clip_class_{i}.npy", text_features)

