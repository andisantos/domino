#!/usr/bin/env python
# coding: utf-8



from dataloader_places import PlacesDataset
from dataloader_pacs import Dataloader_PACS
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import pandas as pd
from PIL import Image 
import numpy as np
from plot_mosaic import PlotMosaic

plot = PlotMosaic(img_size=64)

# # PACS
# dataset = Dataloader_PACS()
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
# datasetname = 'PACS'

# PLACES 8
idx_to_classname ={0: 'bathroom', 1: 'bedroom', 2: 'childs_room',
                   3: 'classroom', 4: 'dressing_room', 5: 'living_room',
                   6: 'studio', 7: "swimming_pool"}

data_path = "../adversarial-sets/data/Places8_paths_and_labels_complete_train.npy"

places_ds = [PlacesDataset(data_path,
                           onlylabels=[k]) for k in range(8)]
batch_size = 1
train_dataloaders_class = {k: DataLoader(places_ds[k],
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=8) for k in range(8)}
for k in range(8):
    print(f"\nDataloader: {batch_size} batch size | {len(train_dataloaders_class[k])} batches | {len(train_dataloaders_class[k].dataset)} images")


# Places8
csvs = ['outputs/class0_bathroom_2slices_binarytargets.csv', 'outputs/class1_bedroom_2slices_binarytargets.csv',
        'outputs/class2_childsroom_2slices_binarytargets.csv', 'outputs/class3_classroom_2slices_binarytargets.csv',
        'outputs/class4_dressingroom_2slices_binarytargets.csv', 'outputs/class5_livingroom_2slices_binarytargets.csv',
       'outputs/class6_studio_2slices_binarytargets.csv', 'outputs/class7_swimmingpool_2slices_binarytargets.csv']

# PACS
# csvs = ['outputs/PACS_class0_4slices_binarytargets.csv', 'outputs/PACS_class1_4slices_binarytargets.csv',
#        'outputs/PACS_class2_4slices_binarytargets.csv', 'outputs/PACS_class3_4slices_binarytargets.csv',
#        'outputs/PACS_class4_4slices_binarytargets.csv', 'outputs/PACS_class5_4slices_binarytargets.csv',
#        'outputs/PACS_class6_4slices_binarytargets.csv']


for idx, csv_file in enumerate(csvs):
    df = pd.read_csv(csv_file, index_col=[0])
    df = df.iloc[:, 0]
    print(len(df))



# for idx, csv_file in enumerate(csvs):
#     df = pd.read_csv(csv_file, index_col=[0])
#     df = df.iloc[:, 0]
    
#     image_list_cluster0 = []
#     image_list_cluster1 = []

#     for idx2, element in enumerate(train_dataloaders_class[idx]):
#         if df[idx2] == 0:
#             image_list_cluster1.append(plot.img_reshape(element[2][0], 28))
#         else: 
#             image_list_cluster0.append(plot.img_reshape(element[2][0], 28))
    
#     for img_cluster in [(image_list_cluster0, 0), (image_list_cluster1, 1)]:
#         df_images = plot.img_list_to_df(img_cluster[0][:10000])
#         png_name = csv_file.split('_')[0] + f'_{img_cluster[1]}.png'
#         plot.df_to_img_mosaic(df_images, png_name, bmnist = False, img_names = False)
    

npys_train = []
npys_test = []
    

for idx, csv_file in enumerate(csvs):
    image_list_cluster0 = []
    image_list_cluster1 = []
    print(f"Processing npy {csv_file}")
    df = pd.read_csv(csv_file, index_col=[0])
    df = df.iloc[:, 0]

    for idx2, element in enumerate(train_dataloaders_class[idx]):
        if df[idx2] == 0: #coluna do cluster 0
            image_list_cluster1.append((element[2][0], int(element[1].cpu())))
        else: 
            image_list_cluster0.append((element[2][0], int(element[1].cpu())))
    
    print(len(image_list_cluster0), len(image_list_cluster1))
    print((len(image_list_cluster0) + len(image_list_cluster1)))
    if len(image_list_cluster0) > len(image_list_cluster1):
        npys_train = npys_train + image_list_cluster0
        npys_test = npys_test + image_list_cluster1
    else:
        npys_train = npys_train + image_list_cluster1
        npys_test = npys_test + image_list_cluster0


print(len(npys_train), len(npys_test))


np.save('outputs/places8_domino_testset.npy', npys_test)
np.save('outputs/places8_domino_trainset.npy', npys_train)


# # PACS
# for idx, csv_file in enumerate(csvs):
#     df = pd.read_csv(csv_file, index_col=[0])
#     df_numpy = df.to_numpy()
#     image_list_cluster0 = []
#     image_list_cluster1 = []
#     image_list_cluster2 = []
#     image_list_cluster3 = []
#     counter = 0
#     for idx2, element in enumerate(dataloader): # element = img, label, img_path, domain
#         label = element[1]
#         if idx == int(label):
#             # index of the cluster column
#             selected_cluster = int(np.where(df_numpy[counter] == 1)[0])
#             if selected_cluster == 0: # belongs to cluster 0
#                 image_list_cluster0.append(plot.img_reshape(element[2][0], 64)) # imgpath, img_size
#             elif selected_cluster ==1:
#                 image_list_cluster1.append(plot.img_reshape(element[2][0], 64)) # imgpath, img_size
#             elif selected_cluster ==2:
#                 image_list_cluster2.append(plot.img_reshape(element[2][0], 64)) # imgpath, img_size
#             else: 
#                 image_list_cluster3.append(plot.img_reshape(element[2][0], 64)) # imgpath, img_size
#             counter += 1
    
#     for img_cluster in [(image_list_cluster0, 0), (image_list_cluster1,1),
#                                        (image_list_cluster2,2), (image_list_cluster3,3)]:
#         df_images = plot.img_list_to_df(img_cluster[0])
#         png_name = f'outputs/{datasetname}_' + csv_file.split('_')[1] + f'_cluster{img_cluster[1]}.png'
#         plot.df_to_img_mosaic(df_images, png_name, bmnist = False, img_names = False)
