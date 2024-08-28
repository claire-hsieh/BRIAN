import os
import torch
import h5py
import re
import numpy as np
from sklearn.cluster import HDBSCAN, DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import regex as re
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import json
import sys
import time

# Loading and saving embeddings
def save_embedding(input_embeddings, output_file):
    tensors = {}    
    for root, dirs, files in os.walk(input_embeddings, topdown=True):
        for name in files:
            data = torch.load(os.path.join(root, name))
            tensors[re.findall(r"UP.*", root)[0] + "/" + name] = data['mean_representations'][6]
    print(f"Loaded embeddings from {input_embeddings}") 

    with h5py.File(output_file, 'w') as f:
        for key, tensor in tensors.items():
            f.create_dataset(key, data=tensor.numpy())
    print(f"Saved embeddings to {output_file}")
    return tensors

def load_embeddings(input_file):
    tensors = {}
    with h5py.File(input_file, 'r') as f:
        def recursively_load_group(group, prefix=''):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Group):
                    recursively_load_group(item, prefix + key + '/')
                else:
                    tensors[prefix + key] = torch.tensor(item[()])

        recursively_load_group(f)
    return tensors


# For visualization / testing
def plot_clusters(cluster, alg_title):
    plt.figure(figsize=(10, 7))
    scatter_handles = []
    labels = []
    for label, i in cluster.items():
        scatter = plt.scatter(i + np.random.uniform(-0.05, 0.05), 0, label=label)
        scatter_handles.append(scatter)
        labels.append(label)
    plt.xlabel('Cluster')
    plt.title(f'{alg_title} Clusters (jittered)')
    plt.show()

    # Create a separate figure for the legend
    fig_legend = plt.figure(figsize=(10, 2))
    fig_legend.legend(handles=scatter_handles, labels=labels, loc='center', ncol=5)
    fig_legend.suptitle('Legend')
    plt.axis('off')
    plt.show()

def plot_pca(tensors, cluster, alg_title):
    labels = [i.split('|')[1] for i in tensors.keys()]
    tensor_list = list(tensors.values())

    # Stack the tensors into a single array
    tensor_array = np.stack(tensor_list)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(tensor_array)

    # Map cluster labels
    cluster_labels = [cluster[label] for label in labels]

    # Plot the PCA results with cluster labels
    plt.figure(figsize=(10, 7))
    scatter_handles = []
    unique_labels = set(cluster_labels)
    for label in unique_labels:
        indices = [i for i, l in enumerate(cluster_labels) if l == label]
        scatter = plt.scatter(pca_result[indices, 0], pca_result[indices, 1], label=label)
        scatter_handles.append(scatter)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'PCA with {alg_title} Clustering')
    plt.show()

    # Create a separate figure for the legend
    fig_legend = plt.figure(figsize=(10, 2))
    fig_legend.legend(handles=scatter_handles, labels=unique_labels, loc='center', ncol=5)
    fig_legend.suptitle('Legend')
    plt.axis('off')
    plt.show()

# Clustering
def hierarchical(tensors):
    flattened_tensors = [tensor.flatten() for tensor in tensors.values()]
    distance_matrix = pdist(flattened_tensors, metric='euclidean')
    linked = linkage(distance_matrix, method='ward')
    t = 4
    clusters = fcluster(linked, t=t, criterion='distance')
    clusters_dict = {}
    for i, label in zip(clusters, tensors.keys()):
        clusters_dict[label.split('|')[1]] = i
    return {k: v for k, v in sorted(clusters_dict.items(), key=lambda item: item[1])}

def dbscan(tensors, eps=2, min_samples=2, alg='auto'):
    tensor_list = list(tensors.values())
    tensor_array = np.stack(tensor_list)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(tensor_array)
    result_dict = {key: label for key, label in zip(tensors.keys(), labels)}
    return result_dict


def hdbscan(tensors, min_samples=2, min_cluster_size=5):
    tensor_list = list(tensors.values())
    tensor_array = np.stack(tensor_list)
    dbscan = HDBSCAN(min_samples, min_cluster_size)
    labels = dbscan.fit_predict(tensor_array)
    result_dict = {key: label for key, label in zip(tensors.keys(), labels)}

    return result_dict


if __name__ == '__main__':
    # Parameters
    testing = False
    if testing:
        alg = 'hdbscan'
        save_embeddings = True
        if save_embeddings: input_dir = '/home/gluetown/brain/test_set/samples/'
        else: input_file = '/home/gluetown/brain/test_set/test/test_embeddings.h5'
        output_embeddings = '/home/gluetown/brain/test_set/test/test_embeddings.h5' # must be h5
        output_clusters = '/home/gluetown/brain/test_set/test/test_clusters.txt'


    else: # set parameteres here!!!
        alg = 'hdbscan'
        save_embeddings = True
        if save_embeddings: input_dir = '/home/gluetown/brain/final_embeddings/'
        else: input_file = '/home/gluetown/brain/clusters/final_embeddings.h5'
        output_embeddings = '/home/gluetown/brain/clusters/final_embeddings.h5'
        output_clusters = f'/home/gluetown/brain/clusters/{alg}_clusters_1.txt'

    print("Imported Libraries")
    if save_embeddings:
        time1 = time.time()
        tensors = save_embedding(input_dir, output_embeddings)
        time2 = time.time()
        print(f"Loaded embeddings from {input_dir}")
    else:
        time1 = time.time()
        tensors = load_embeddings(input_file)
        time2 = time.time()
        print(f"Loaded embeddings from {input_file}")

    print(f"Number of embeddings: {len(tensors)}")
    print(f"Time to save embeddings: {time2 - time1} seconds")

    if alg == 'hierarchical':
        time3 = time.time()
        test_clusters = hierarchical(tensors)
        time4 = time.time()
        print(f"Time to cluster with {alg}: {time4 - time3} seconds")
        with open(output_clusters, 'w') as f:
            for key, value in test_clusters.items():
                f.write(f"{key},{value}\n")

    elif alg == 'dbscan':
        time3 = time.time()
        clusters = dbscan(tensors, eps=3, min_samples=5)
        dbscan_clusters = {i.split('|')[1]: label for i, label in clusters.items()}
        time4 = time.time()
        print(f"Time to cluster with {alg}: {time4 - time3} seconds")
        del clusters
        with open(output_clusters, 'w') as f:
            for key, value in dbscan_clusters.items():
                f.write(f"{key},{value}\n")

    elif alg == 'hdbscan':
        time3 = time.time()
        clusters = hdbscan(tensors, min_cluster_size=2)
        hdbscan_clusters = {i.split('|')[1]: label for i, label in clusters.items()}
        time4 = time.time()
        print(f"Time to cluster with {alg}: {time4 - time3} seconds")
        del clusters
        with open(output_clusters, 'w') as f:
            for key, value in hdbscan_clusters.items():
                f.write(f"{key},{value}\n")

