import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler


# Récupération des données
jain=pd.read_csv("./data/jain.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"]).to_numpy()
aggregation=pd.read_csv("./data/aggregation.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"]).to_numpy()
pathbased=pd.read_csv("./data/pathbased.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"]).to_numpy()

# Pré-traitement des données
scaler=StandardScaler(with_std=True) # True au lieu de False pour réduire les données
Z_jain=scaler.fit_transform(jain)
Z_aggregation=scaler.fit_transform(aggregation)
Z_pathbased=scaler.fit_transform(pathbased)
dataset=[Z_jain, Z_aggregation, Z_pathbased]

# Fonction de construction d'un CAH
def CAH(data, methode, pas, criter):
    # optimal_ordering : booleen demandant s'il faut, oui ou non, trier les feuilles
    # metric : metrique utilisées pour la mesure des distances
    # method : choix de la méthode de reunion des données (distance de ward, min, max, avg ... entre 2 clusters)

    Z_linked=linkage(data[:, :-1], optimal_ordering=True, method=methode, metric='euclidean')
    # plt.figure(figsize=(10, 7))
    # dendrogram(Z_linked, color_threshold=20) # Pramètres [données linked, paramètres d'affichage]
    # plt.show()
    clusters = fcluster(Z_linked, t=pas, criterion=criter)

    # Affichage des données
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Clusters obtenus avec l'algorithme
    ax[0].scatter(data[:, 0], data[:, 1], c=clusters)
    ax[0].set_title('Clusters CAH')
    ax[0].set_xlabel('1ère caractéristique')
    ax[0].set_ylabel('2ème caractéristique')
    ax[0].grid(True)

    # Résultats attendus
    ax[1].scatter(data[:, 0], data[:, 1], c=data[:,2])
    ax[1].set_title('Clusters réels')
    ax[1].set_xlabel('1ère caractéristique')
    ax[1].set_ylabel('2ème caractéristique')
    ax[1].grid(True)

    plt.show()

    # Evaluation des performances
    print("Le score de DB est :", davies_bouldin_score(data, clusters))
    print("Le score de silhouette est :", silhouette_score(data, labels = clusters, metric='euclidean'))
    print("Le score de rand est :", adjusted_rand_score(data['Résultat'], clusters))
    return silhouette_score(data, labels = clusters, metric='euclidean')

# Trouver la meilleure methode ['single', 'complete', 'average', 'median', 'centroid', 'weighted', 'ward']
method=['ward', 'average']
criterion=['distance', 'maxclust'] # La méthode 'inconsistent' crée des dégradés
t_ward=[16, 7, 9] # pour criterion='distance'
t_avg=[1.8, 1.1, 1.7] # pour criterion='average'

# Trouver t optimal
# L_score=[]
# L_index=[]
# for i in range (10, 20, 1):
#     L_score.append(CAH(dataset[0], method[0], i/10, criterion[0]))
#     L_index.append(i/10)
# plt.plot(L_index, L_score)
# plt.ylabel('Score de silhouette')
# plt.xlabel('Valeur de t')
# plt.show()

CAH(dataset[0], method[0], t_ward[0], criterion[0])
CAH(dataset[1], method[1], 7, criterion[1])
CAH(dataset[2], method[0], t_ward[2], criterion[0])