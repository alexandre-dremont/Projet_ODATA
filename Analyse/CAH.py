from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, fcluster

# Fonction de construction d'un CAH
def CAH(data, methode, pas, criter):
    Z_linked=linkage(data[:, :-1], optimal_ordering=True, method=methode, metric='euclidean')
    clusters = fcluster(Z_linked, t=pas, criterion=criter)

    # Evaluation des performances
    print("Le score de DB est :", davies_bouldin_score(data, clusters))
    print("Le score de silhouette est :", silhouette_score(data, labels = clusters, metric='euclidean'))
    return clusters

