from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import silhouette_score, adjusted_rand_score

jain=pd.read_csv("./data/jain.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"])
# print(jain)

aggregation=pd.read_csv("./data/aggregation.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"])

pathbased=pd.read_csv("./data/pathbased.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"])

def k_means(X, n_clusters, init = 'k-means++', n_init=20, max_iter=100):
    # Nombre de clusters = valeur maximale du résultat dans le fichier .txt (np.max(X["Résultat"]))
    start_time=time.time()
    kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter).fit(X.drop(columns=["Résultat"]))
    end_time=time.time()
    # print(kmeans.labels_)
    # print(kmeans.predict([[0, 0], [12, 3]]))
    # print(kmeans.cluster_centers_)
    # print(kmeans.inertia_) # Inertie intra-classe
    # print(kmeans.n_iter_) # Nombre d'itérations
    # print(kmeans.n_features_in_) # Nombre de colonnes prises en compte pour le clustering
    # print(kmeans.feature_names_in_) # Colonnes prises en compte pour le clustering

    labels=kmeans.labels_ 

    # clusters=[[] for i in range (max(labels)+1)]
    # for i in range (len(labels)): # for i in enumerate
    #     clusters[labels[i]].append(i)
    # print(clusters)

    # print(kmeans.score(X.drop(columns=["Résultat"]),X["Résultat"]))

    # Affichage des données
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Clusters obtenus avec l'algorithme
    ax[0].scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
    ax[0].set_title('Clusters K-means')
    ax[0].set_xlabel('1ère caractéristique')
    ax[0].set_ylabel('2ème caractéristique')
    ax[0].grid(True)

    # Résultats attendus
    ax[1].scatter(X.iloc[:, 0], X.iloc[:, 1], c=X.iloc[:,2])
    ax[1].set_title('Clusters réels')
    ax[1].set_xlabel('1ère caractéristique')
    ax[1].set_ylabel('2ème caractéristique')
    ax[1].grid(True)

    plt.show()

    return kmeans.inertia_, labels, end_time-start_time

# print(k_means(jain, 2))
# print(k_means(aggregation, 4))
# print(k_means(pathbased, 6))

def choix_K(X):
    # Deux façons de justifier le K optimal : tracer le coefficient de silhouette ou l'indice de rand en fonction de k ou tracer l'inertie 
    # intra_classe en fonction k et utiliser le critère du "coude"

    K=[2]+[i for i in range(2,10)]
    I=[]
    sil=[]
    ari=[]
    T=[]
    for k in K:
        inertie, labels, t = k_means(X, k)
        I.append(inertie)
        sil.append(silhouette_score(X, labels = labels, metric='euclidean'))
        ari.append(adjusted_rand_score(X["Résultat"] , labels))
        T.append(t)
    
    K.pop(0)
    I.pop(0)
    sil.pop(0)
    ari.pop(0)
    T.pop(0)

    fig, ax = plt.subplots(1, 3, figsize=(14, 5))

    ax[0].bar(K, I) 
    ax[0].plot(K, I)
    ax[0].set_title('Inertie intra-classe en fonction de K')
    ax[0].set_xlabel('K')
    ax[0].set_ylabel('Inertie intra-classe')

    ax[1].plot(K, sil)
    ax[1].set_title('Coefficient de silhouette en fonction de K')
    ax[1].set_xlabel('K')
    ax[1].set_ylabel('Coefficient de silhouette')

    ax[2].plot(K, ari)
    ax[2].set_title('Coefficient de rand ajusté en fonction de K')
    ax[2].set_xlabel('K')
    ax[2].set_ylabel('Coefficient de rand')
    plt.show()

    plt.plot(K, T)
    plt.title("Temps d'éxécution de l'algorithme en fonction de K")
    plt.xlabel('K')
    plt.ylabel("Temps d'éxécution")
    plt.show()

# choix_K(jain)
# choix_K(aggregation)
# choix_K(pathbased)
    



