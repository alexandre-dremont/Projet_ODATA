from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score


def k_means(X, n_clusters, init, n_init, algorithm, max_iter):
    
    start_time=time.time()
    kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, algorithm=algorithm).fit(X)
    end_time=time.time()
    # print(kmeans.labels_)
    # print(kmeans.predict([[0, 0], [12, 3]]))
    # print(kmeans.cluster_centers_)
    # print(kmeans.inertia_) # Inertie intra-classe
    print("init = ", init)
    print("n_init = ", n_init)
    print("algorith = ", algorithm)
    print("Le nombre d'itérations est", kmeans.n_iter_) # Nombre d'itérations
    # print(kmeans.n_features_in_) # Nombre de colonnes prises en compte pour le clustering
    # print(kmeans.feature_names_in_) # Colonnes prises en compte pour le clustering

    labels=kmeans.labels_ 

    # print(kmeans.score(X.drop(columns=["Résultat"]),X["Résultat"]))

    print("L'inertie est", kmeans.inertia_)
    print("Le coefficient de silhouette est ", silhouette_score(X, labels = labels, metric='euclidean'))
    # print("L'indice de rand ajusté est ", adjusted_rand_score(X["Résultat"], labels))
    print("L'indice de Davies-Bouldin est", davies_bouldin_score(X, labels))
    print("Le temps d'éxécution est", end_time-start_time)

    return kmeans.inertia_, labels, end_time-start_time


def choix_K_kmeans(X, init, n_init, algorithm, max_iter):
    # Deux façons de justifier le K optimal : tracer le coefficient de silhouette ou l'indice de rand en fonction de k ou tracer l'inertie 
    # intra_classe en fonction k et utiliser le critère du "coude"

    K=[2]+[i for i in range(2,10)]
    I=[]
    sil=[]
    ari=[]
    DB=[]
    T=[]
    for k in K:
        inertie, labels, t = k_means(X, k, init, n_init, algorithm, max_iter)
        I.append(inertie)
        sil.append(silhouette_score(X, labels = labels, metric='euclidean'))
        # ari.append(adjusted_rand_score(X["Résultat"] , labels))
        T.append(t)
        DB.append(davies_bouldin_score(X, labels))
    
    K.pop(0)
    I.pop(0)
    sil.pop(0)
    # ari.pop(0)
    T.pop(0)
    DB.pop(0)

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

    ax[2].plot(K, DB)
    ax[2].set_title("Score de Davies-Bouldin en fonction de K")
    ax[2].set_xlabel('K')
    ax[2].set_ylabel("Score de Davies-Bouldin")
    plt.show()

    # plt.plot(K, ari)
    # plt.title("Coefficient de rand en fonction de K")
    # plt.xlabel('K')
    # plt.ylabel("Coefficient de rand")
    # plt.show()

    # plt.plot(K, T)
    # plt.title("Temps d'éxécution de l'algorithmeCoefficient de rand ajusté en fonction de K")
    # plt.xlabel('K')
    # plt.ylabel("Temps d'éxécution de l'algorithme")
    # plt.show()
    



