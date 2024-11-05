import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from sklearn.cluster import SpectralClustering

# Récupération des données

jain=pd.read_csv("./data/jain.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"])

aggregation=pd.read_csv("./data/aggregation.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"])

pathbased=pd.read_csv("./data/pathbased.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"])

# Fonction de clustering spectral

def specClust(data, nb_cluster, matrix, sigma=1.0, KNN=10):
    if matrix=='rbf':
        spectre=SpectralClustering(n_clusters=nb_cluster, affinity=matrix, gamma=sigma)
    elif matrix=="nearest_neighbors":
        spectre=SpectralClustering(n_clusters=nb_cluster, affinity=matrix, n_neighbors=KNN)
    else :
        spectre=SpectralClustering(n_clusters=nb_cluster, affinity=matrix)
    labels = spectre.fit_predict(data)

    # Affichage des données
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Clusters obtenus avec l'algorithme
    ax[0].scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels)
    ax[0].set_title('Clustering Spectral')
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

    DB = davies_bouldin_score(data, labels)
    sil = silhouette_score(data, labels, metric='euclidean')
    rand = adjusted_rand_score(data['Résultat'], labels)

    print("Le score de DB est :", DB)
    print("Le score de silhouette est :", sil)
    print("Le score de Rand est :", rand)
    return [sil, rand, DB]

specClust(jain, 2, 'rbf')
specClust(aggregation, 7, 'rbf')
specClust(pathbased, 3, 'rbf')