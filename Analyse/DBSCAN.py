import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score

# Fonction de DBSCAN
def DBSCAN_clust(data, k=4):

    # Calcul du epsilon optimal
    voisins = NearestNeighbors(n_neighbors=k).fit(data)
    distances, indices = voisins.kneighbors(data)
    k_distances = np.sort(distances[:, k-1])[::-1]
    epsilon = k_distances[np.argmax(np.diff(k_distances))]

    # Entrainement du modèle 
    density=DBSCAN(eps=1.12*epsilon, min_samples=2*len(data[0]))
    labels=density.fit_predict(data)

    # Evaluation des performances
    DB = davies_bouldin_score(data, labels)
    sil = silhouette_score(data, labels, metric='euclidean')

    print("Le score de DB est :", DB)
    print("Le score de silhouette est :", sil)
    return labels