#!/usr/bin/env python
# coding: utf-8
import torchvision

from dataloader_places import PlacesDataset
from torch.utils.data import DataLoader
# from domino._embed import embed
import clip
import torch
from tqdm import tqdm
import torch.nn.functional as nnf
import numpy as np

idx_to_classname ={0: 'bathroom', 1: 'bedroom', 2: 'childs_room',
                   3: 'classroom', 4: 'dressing_room', 5: 'living_room',
                   6: 'studio', 7: "swimming_pool"}

data_path = "../adversarial-sets/data/Places8_paths_and_labels_complete_train.npy"
dataset = PlacesDataset(data_path)
batch_size = 64
dataloader = DataLoader(dataset, shuffle=False, num_workers=6, batch_size=batch_size)
print(f"\nDataloader: {batch_size} batch size | {len(dataloader)} batches | {len(dataloader.dataset)} images")

print(clip.available_models())
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model_clip, preprocess = clip.load("ViT-B/32", device=device)
print(preprocess)

model_path = "../adversarial-sets/outputs/resnet50_mit-sun/resnet50_mit-sun.pth"
model_resnet50 = torchvision.models.resnet50()
model_resnet50.fc = torch.nn.Linear(in_features=2048, out_features=8, bias=True)
model_resnet50.load_state_dict(torch.load(model_path))
model_resnet50.to(device)
model_resnet50.eval()


image_features = []
text_features = []
image_resnet50_outputs = []
gt = []
for inputs, labels, _ in tqdm(dataloader):
    inputs = inputs.to(device)
    text_inputs = torch.cat([clip.tokenize(
        f"a photo of a {idx_to_classname[label.item()]}", truncate=True) for label in labels]).to(device)
    labels = labels.to(device)
    
    # Calculate features
    with torch.no_grad():
        image_features = image_features + list(model_clip.encode_image(inputs).cpu().data.numpy())
        text_features = text_features + list(model_clip.encode_text(text_inputs).cpu().data.numpy())
        
        #resnet50 probability prediction
        outputs = model_resnet50(inputs)
        outputs = nnf.softmax(outputs, dim=1)
        top_p, top_class = outputs.topk(1, dim = 1)
        image_resnet50_outputs = image_resnet50_outputs + list(top_class.cpu().data.numpy())
        gt = gt + list(labels.cpu().data.numpy())

image_features = np.asarray(image_features)
print("Clip image features shape: ", image_features.shape)
text_features = np.asarray(text_features)
print("Clip text features shape: ", text_features.shape)
gt = np.asarray(gt)
print("Targets shape: ", gt.shape)

image_resnet50_outputs = np.asarray(image_resnet50_outputs)
image_resnet50_outputs = nnf.one_hot(torch.tensor(image_resnet50_outputs.squeeze()))
image_resnet50_outputs = image_resnet50_outputs.numpy()
print("Preds 1 hot encoded shape:", image_resnet50_outputs.shape)

print("Saving npys...")
np.save("places8_image_features_clip.npy", image_features)
np.save("places8_text_features_clip.npy", text_features)
np.save("places8_image_targets.npy", gt)
np.save("places8_image_preds_1hot.npy", image_resnet50_outputs)
print("Done!")
