from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import SpectralClustering

# Fonction de clustering spectral
def specClust(data, nb_cluster, matrix, sigma=1.0, KNN=10):
    if matrix=='rbf':
        spectre=SpectralClustering(n_clusters=nb_cluster, affinity=matrix, gamma=sigma)
    elif matrix=="nearest_neighbors":
        spectre=SpectralClustering(n_clusters=nb_cluster, affinity=matrix, n_neighbors=KNN)
    else :
        spectre=SpectralClustering(n_clusters=nb_cluster, affinity=matrix)
    labels = spectre.fit_predict(data)

    # Evaluation des performances
    DB = davies_bouldin_score(data, labels)
    sil = silhouette_score(data, labels, metric='euclidean')

    print("Le score de DB est :", DB)
    print("Le score de silhouette est :", sil)
    return labels