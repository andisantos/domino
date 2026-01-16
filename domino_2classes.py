#!/usr/bin/env python
# coding: utf-8


from domino import DominoSlicer
import numpy as np
import sklearn
import pandas as pd
sklearn.__version__

import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
import torch
import os
from time import time

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
outpath = os.path.join("outputs", "places2")
os.makedirs(outpath, mode=777, exist_ok=True)


def gen_tsne(predict, clip_emb, plot_n = None, n = 5, pdf_name=None):
    cluster_label = np.argmax(predict, axis = -1)
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
                                   'clusterlabel': filtered_labels})
    if pdf_name:
        tsne_result_df.to_csv(pdf_name, index=False)
    #fig, ax = plt.subplots(1)
    #sns.scatterplot(x = 'component1',
    #                y = 'component2',
    #                hue = 'label',
    #                data = tsne_result_df, ax = ax, sizes = 5)
    #lim = (X_embedded.min() - 5, X_embedded.max() + 5)
    #ax.set_xlim(lim)
    #ax.set_ylim(lim)
    #ax.set_aspect('equal')
    #ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.0)


# ## 2 classes | 3 slices

slicer = DominoSlicer(
    y_log_likelihood_weight=10,
    y_hat_log_likelihood_weight=10,
    n_mixture_components=100,
    n_slices=3,
    confusion_noise= 0.001
    )
print(slicer.get_params())


# ## class 1: bedroom | 3 slices
places8_clip_emd = "data/image_features_clip_class_1.npy"  # embeds created for 8 classes
places8_image_softmax = "data/features_resnet50_class_0.npy" # bedroom
places8_targets = "data/places8_image_targets.npy" #created for 2 classes
clip_emb_1 = np.load(places8_clip_emd)
preds_softmax_1 = np.load(places8_image_softmax)
places8_targets = np.load(places8_targets)
places8_targets_1 = np.zeros(places8_targets[np.where(places8_targets==0)[0]].shape, dtype=np.uint8)

# filter softmax per index 1
print("Data with class 0 (bedroom):")
print(f"Clip embed shape {clip_emb_1.shape} | targets shape {places8_targets_1.shape} | preds softmax shape {preds_softmax_1.shape}")

start_time = time()
run_id = 0
executions = 0
while run_id <= 5:
    executions +=1
    _ = slicer.fit(embeddings = clip_emb_1, targets = places8_targets_1, pred_probs = preds_softmax_1)
    predict = slicer.predict(embeddings = clip_emb_1, targets = places8_targets_1, pred_probs = preds_softmax_1)
    
    cluster_label = np.argmax(predict, axis = -1)
    label, counts = np.unique(cluster_label, return_counts = True)
    print(label, counts)
    counts_min = min(counts)
    counts_max = max(counts)
    # if counts_min >= 5000 and counts_max <=60000:
    if counts_min >= int(clip_emb_1.shape[0]*0.10) and counts_max <= clip_emb_1.shape[0]*0.65:
        print(f"Saving! {run_id}")
        gen_tsne(predict, clip_emb_1, n=3, pdf_name=f"{outpath}/tsne_components_bedroom_run_{run_id}.csv")
        df = pd.DataFrame(predict, columns=['group_0', 'group_1', 'group_2'])
        print(len(df))
        df.to_csv(f"{outpath}/bedroom_3slices_{run_id}.csv", index = False, encoding='utf-8')
        run_id += 1

total = time() - start_time

print("Total executions for class BEDROOM:", executions, "with",  total/60, "minutes" )


# ## class 2: child's room | 3 slices

places8_clip_emd = "data/places8_image_features_clip_class_2.npy" # embeds created for 8 classes
places8_image_softmax = "data/features_resnet50_class_1.npy" # childsroom
places8_targets = "data/places8_image_targets.npy" #created for 2 classes

places8_targets = np.load(places8_targets)
clip_emb_2 = np.load(places8_clip_emd)
preds_softmax_2 = np.load(places8_image_softmax)
places8_targets_2 = np.ones(places8_targets[np.where(places8_targets == 1)[0]].shape, dtype=np.uint8)

# filter softmax per index 2
print("Data with class 1 (childsroom")
print(f"Clip embed shape {clip_emb_2.shape} | target shape {places8_targets_2.shape} | preds softmax shape {preds_softmax_2.shape}")

start_time = time()
run_id = 0
executions = 0
while run_id <= 5:
    executions+=1
    _ = slicer.fit(embeddings=clip_emb_2, targets =places8_targets_2, pred_probs=preds_softmax_2)
    predict = slicer.predict(embeddings=clip_emb_2, targets=places8_targets_2, pred_probs=preds_softmax_2)
    
    cluster_label = np.argmax(predict, axis=-1)
    label, counts = np.unique(cluster_label, return_counts = True)
    print(label, counts)
    counts_min = min(counts)
    counts_max = max(counts)
    if counts_min >= clip_emb_2.shape[0]*0.15:
        print(f"Saving! {run_id}")
        gen_tsne(predict, clip_emb_1, n=3, pdf_name=f"{outpath}/tsne_components_childsroom_run_{run_id}.csv")
        df = pd.DataFrame(predict, columns=['group_0', 'group_1', 'group_2'])
        print(len(df))
        df.to_csv(f"{outpath}/childsroom_3slices_{run_id}.csv", index=False, encoding='utf-8')
        run_id += 1

total = time() - start_time
print("Total executions for class CHILDSROOM:", executions, "with",  total/60, "minutes" )