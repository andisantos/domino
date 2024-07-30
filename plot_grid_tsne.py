#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install matplotlib


# In[2]:


from sklearn.manifold import TSNE
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from dataloader_places_diff import PlacesDataset
from PIL import Image


# In[3]:


def gen_tsne(predict, clip_emb, plot_n = None, n = 3):
    cluster_label = np.argmax(predict, axis = -1)
    print(cluster_label.shape)
    label, counts = np.unique(cluster_label, return_counts = True)
    print(label, counts)
    counts = min(counts)

    if plot_n != None:
            counts = plot_n
    random.seed(42)
    
    filter_0 = random.sample(list(np.where(cluster_label == 0)[0]), counts)
    filter_1 = random.sample(list(np.where(cluster_label == 1)[0]), counts)
    filtered_clip_embeds = clip_emb[filter_0]
    filtered_labels = cluster_label[filter_0]
    filtered_clip_embeds = np.concatenate((filtered_clip_embeds, clip_emb[filter_1]))
    filtered_labels = np.concatenate((filtered_labels, cluster_label[filter_1]))
    if n == 3:
        filter_2 = random.sample(list(np.where(cluster_label == 2)[0]), counts)
        filtered_clip_embeds = np.concatenate((filtered_clip_embeds, clip_emb[filter_2]))
        filtered_labels = np.concatenate((filtered_labels, cluster_label[filter_2]))
    X_embedded = TSNE(n_components=2, learning_rate='auto', 
                      init='random', perplexity=3).fit_transform(filtered_clip_embeds)
    X_embedded.shape
    tsne_result_df = pd.DataFrame({'component1': X_embedded[:,0],
                                   'component2': X_embedded[:,1],
                                   'label': filtered_labels})
    tsne_result_df.to_csv('outputs/plot_tsne/tsne_components-childs_room.csv', index=False)
    return X_embedded


# In[4]:


clip_emb_1 = np.load("data/places8_image_features_clip_class_1.npy") #bedroom
# df = pd.read_csv("outputs/places8_bedroom_childs_room/class1_bedroom_3slices_otherclassessoftmax.csv", index_col=0)

clip_emb_2 = np.load("data/places8_image_features_clip_class_2.npy") #childs room
# df = pd.read_csv("outputs/places8_bedroom_childs_room/class2_childs_room_3slices_otherclassessoftmax.csv", index_col=0)

clip_embeds = np.concatenate([clip_emb_1, clip_emb_2])
df = pd.read_csv("outputs/places8_bedroom_childs_room/class2_childs_room_3slices_2classes_softmax.csv")
tsne_results = gen_tsne(df.to_numpy(), clip_embeds, n = 3)


# In[5]:


dataset = PlacesDataset("data/Places8_paths_and_labels_complete_train.npy", onlylabels=[1,2])
dataloader = dataset.get_loader(False, 128, 4)
imgs_bedroom = []
for [imgs_0, _, _] in dataloader:
    imgs_0 = imgs_0.cpu().data.numpy()
    imgs_bedroom += list(imgs_0)

imgs_bedroom = np.asarray(imgs_bedroom)
print(imgs_bedroom.shape)
imgs_bedroom_out = [np.moveaxis(x, 0, -1) for x in imgs_bedroom]
imgs_bedroom_out = np.asarray(imgs_bedroom_out)
print(imgs_bedroom_out.shape)


# In[ ]:

print("calculating cost matrix...")
sel_tsne = tsne_results
out_dim = int(np.ceil(np.sqrt(sel_tsne.shape[0])))
out_res = 64
# rows = int(np.ceil(x_data.shape[0] * 1. / cols))
grid = np.dstack(np.meshgrid(np.linspace(0, 1, out_dim), np.linspace(0, 1, out_dim))).reshape(-1, 2)
cost_matrix = cdist(grid, sel_tsne, "sqeuclidean").astype(np.float32)
cost_matrix = cost_matrix * (100000 / cost_matrix.max())

row_asses, col_asses = linear_sum_assignment(cost_matrix)
print("done")

# In[ ]:


grid_jv = grid[col_asses]
out = np.ones((out_dim * out_res, out_dim * out_res, 3))
print("plotting images...")
for pos, img in zip(grid_jv, imgs_bedroom_out):
    h_range = int(np.floor(pos[0] * (out_dim - 1) * out_res))
    w_range = int(np.floor(pos[1] * (out_dim - 1) * out_res))
    out[h_range:h_range + out_res, w_range:w_range + out_res] = img

im = Image.fromarray(np.uint8(out * 255))
image_grid_plot_file = 'outputs/plot_tsne/tsne-image-grid-plot-class_childs_room.png'
im.save(image_grid_plot_file)
print("Done!")



