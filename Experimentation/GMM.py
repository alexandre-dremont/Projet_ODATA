from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score

# 3ème méthode de Clustering : modèle de mélange de Gaussiennes

jain=pd.read_csv("./data/jain.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"])

aggregation=pd.read_csv("./data/aggregation.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"])

pathbased=pd.read_csv("./data/pathbased.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"])

def GMM(X, n_components, covariance_type, n_init = 10, max_iter=1000):

    start_time = time.time()
    gm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, n_init = n_init, max_iter=max_iter).fit(X.drop(columns=["Résultat"]))
    end_time = time.time()

    # print(gm.weights_) # poids des composantes gaussiennes 
    # print(gm.means_)
    # print(gm.covariances_) 
    # print(gm.precisions_) # Matrice de précision (inverse de la covariance) pour chaque composante gaussienne
    # print(gm.precisions_cholesky_) # Décomposition de Cholesky des matrices de précision, utilisée pour évaluer plus rapidement les densités de probabilité des points
    # print(gm.converged_)
    # print(gm.n_iter_)
    # print(gm.lower_bound_)
    # print(gm.n_features_in_)
    # print(gm.feature_names_in_)
    # print(gm.predict([[0, 0], [12, 3]]))
    # print(gm.bic(X))
    # print(gm.aic(X))

    P=gm.predict_proba(X.drop(columns=["Résultat"]))
    # print(gm.aic(X))
    # print(gm.bic(X))
    labels_predicted=[np.argmax(P[i]) for i in range (len(P))]
    # print(len(labels_predicted))

    # Affichage des données
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Clusters obtenus avec l'algorithme
    ax[0].scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels_predicted)
    ax[0].set_title('Clusters Mélange de Gaussiennes')
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

    print(silhouette_score(X.drop(columns=["Résultat"]), labels = labels_predicted, metric='euclidean'))
    print(adjusted_rand_score(X["Résultat"], labels_predicted))
    print(davies_bouldin_score(X.drop(columns=["Résultat"]), labels_predicted))
    

    return labels_predicted, end_time-start_time

GMM(jain, 2, 'full')
# GMM(jain, 2, 'tied')
# GMM(jain, 2, 'diag')
# GMM(jain, 2, 'spherical')
# print("_________________")
# GMM(aggregation, 7, 'full')
# GMM(aggregation, 7, 'tied')
# GMM(aggregation, 7, 'diag')
# GMM(aggregation, 7, 'spherical')
# print("_________________")
# GMM(pathbased, 3, 'full')
# GMM(pathbased, 3, 'tied')
# GMM(pathbased, 3, 'diag')
# GMM(pathbased, 3, 'spherical')


def choix_K(X, covariance_type):
    # Deux façons de justifier le K optimal : tracer le coefficient de silhouette ou l'indice de rand en fonction de k ou tracer l'inertie 
    # intra_classe en fonction k et utiliser le critère du "coude"

    K=[2]+[i for i in range(2,10)]
    sil=[]
    ari=[]
    T=[]
    for k in K:
        labels, t = GMM(X, k, covariance_type)
        sil.append(silhouette_score(X, labels = labels, metric='euclidean'))
        ari.append(adjusted_rand_score(X["Résultat"] , labels))
        T.append(t)
    
    K.pop(0)
    sil.pop(0)
    ari.pop(0)
    T.pop(0)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(K, sil)
    ax[0].set_title('Coefficient de silhouette en fonction du nombre de composantes')
    ax[0].set_xlabel('Nombre de composantes')
    ax[0].set_ylabel('Coefficient de silhouette')

    ax[1].plot(K, ari)
    ax[1].set_title('Coefficient de rand ajusté en fonction du nombre de composantes')
    ax[1].set_xlabel('Nombre de composantes')
    ax[1].set_ylabel('Coefficient de rand')
    plt.show()

    # plt.plot(K, T)
    # plt.title("Temps d'éxécution de l'algorithme en fonction du nombre de composantes")
    # plt.xlabel('Nombre de composantes')
    # plt.ylabel("Temps d'éxécution")
    # plt.show()

# choix_K(jain, 'full') # 3 et 5
# choix_K(jain, 'tied') # 2 et 7
# choix_K(jain, 'diag') # 2 et 3
# choix_K(jain, 'spherical') # 2 et 8
# choix_K(aggregation, 'full') # 3, 5 et 7
# choix_K(aggregation, 'tied') # 3 et 5
# choix_K(aggregation, 'diag') # 4 et 6
# choix_K(aggregation, 'spherical') # 5 et 6
# choix_K(pathbased, 'full') # 3 et 6
# choix_K(pathbased, 'tied') # 3 et 5
# choix_K(pathbased, 'diag') # 3 et 8
# choix_K(pathbased, 'spherical') # 3 et 8    