import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score

# Récupération des données

jain=pd.read_csv("./data/jain.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"])
aggregation=pd.read_csv("./data/aggregation.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"])
pathbased=pd.read_csv("./data/pathbased.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"])

# Fonction de DBSCAN
def DBSCAN_clust(data, k=4):

    # Conversion des données
    X=data.iloc[:, :-1].to_numpy()

    # Calcul du epsilon optimal
    voisins = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = voisins.kneighbors(X)
    k_distances = np.sort(distances[:, k-1])[::-1]
    epsilon = k_distances[np.argmax(np.diff(k_distances))]

    # Etrainement du modèle 
    density=DBSCAN(eps=1.12*epsilon, min_samples=2*len(data.to_numpy()[:,:-1][0]))
    labels=density.fit_predict(data)

    # Affichage des données
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Clusters obtenus avec l'algorithme
    ax[0].scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels)
    ax[0].set_title('DBSCAN')
    ax[0].set_xlabel('1ère caractéristique')
    ax[0].set_ylabel('2ème caractéristique')
    ax[0].grid(True)

    # Résultats attendus
    ax[1].scatter(data.iloc[:, 0], data.iloc[:, 1], c=data.iloc[:,2])
    ax[1].set_title('Clusters réels')
    ax[1].set_xlabel('1ère caractéristique')
    ax[1].set_ylabel('2ème caractéristique')
    ax[1].grid(True)

    plt.show()

    # Evaluation des performances
    DB = davies_bouldin_score(data, labels)
    sil = silhouette_score(data, labels, metric='euclidean')
    rand = adjusted_rand_score(data['Résultat'], labels)

    print("Le score de DB est :", DB)
    print("Le score de silhouette est :", sil)
    print("Le score de Rand est :", rand)
    return [sil, rand, DB]

DBSCAN_clust(jain)
DBSCAN_clust(aggregation)
DBSCAN_clust(pathbased)