from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

jain=pd.read_csv("./data/jain.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"])
# print(jain)

kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, max_iter=100).fit(jain.drop(columns=["Résultat"]))
# print(kmeans.labels_)
# print(kmeans.predict([[0, 0], [12, 3]]))
# print(kmeans.cluster_centers_)
# print(kmeans.inertia_)
# print(kmeans.n_iter_) # Nombre d'itérations
# print(kmeans.n_features_in_) # Nombre de colonnes prises en compte pour le clustering
# print(kmeans.feature_names_in_) # Colonnes prises en compte pour le clustering

labels=kmeans.labels_ # for i in enumerate

# clusters=[[] for i in range (max(labels)+1)]
# for i in range (len(labels)):
#     clusters[labels[i]].append(i)
# print(clusters)

# Affichage des données
plt.figure(figsize=(8, 6))
colors=['blue', 'green', 'red', 'orange', 'pink', 'purple', 'grey', "black", "yellow"]
for i in range(len(jain)):
    plt.scatter(jain.iloc[i, 1], jain.iloc[i, 2], color=colors[kmeans.labels_[i]])
plt.title('Affichage des données')
plt.xlabel('1ère caractéristique')
plt.ylabel('2ème caractéristique')
plt.grid(True)
plt.show()

