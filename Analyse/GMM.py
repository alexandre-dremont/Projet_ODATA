from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score

def GMM(X, n_components, covariance_type, n_init, max_iter):

    start_time = time.time()
    gm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, n_init = n_init, max_iter=max_iter).fit(X)
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

    P=gm.predict_proba(X)
    # print(gm.aic(X))
    # print(gm.bic(X))
    labels_predicted=np.array([np.argmax(P[i]) for i in range (len(P))])
    # print(len(labels_predicted))

    print("Le coefficient de silhouette est ", silhouette_score(X, labels = labels_predicted, metric='euclidean'))
    # print("L'indice de rand ajusté est ", adjusted_rand_score(X["Résultat"], labels))
    print("L'indice de Davies-Bouldin est", davies_bouldin_score(X, labels_predicted))
    print("Le temps d'éxécution est", end_time-start_time)
    

    return labels_predicted, end_time-start_time


def choix_K_gmm(X, covariance_type, n_init, max_iter):
    # Deux façons de justifier le K optimal : tracer le coefficient de silhouette ou l'indice de rand en fonction de k ou tracer l'inertie 
    # intra_classe en fonction k et utiliser le critère du "coude"

    K=[1]+[i for i in range(1,25)]
    sil=[]
    # ari=[]
    T=[]
    DB=[]
    for k in K:
        labels, t = GMM(X, k, covariance_type, n_init, max_iter)
        sil.append(silhouette_score(X, labels = labels, metric='euclidean'))
        # ari.append(adjusted_rand_score(X["Résultat"] , labels))
        T.append(t)
        DB.append(davies_bouldin_score(X, labels))
    
    K.pop(0)
    sil.pop(0)
    # ari.pop(0)
    T.pop(0)
    DB.pop(0)

    L=np.array([DB[i]-sil[i] for i in range (len(sil))])

    k_opt=L.argsort()
    print("Les k optimaux classés : ", k_opt)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(K, sil)
    ax[0].set_title('Coefficient de silhouette en fonction du nombre de composantes')
    ax[0].set_xlabel('Nombre de composantes')
    ax[0].set_ylabel('Coefficient de silhouette')

    ax[1].plot(K, DB)
    ax[1].set_title('Score de Davies-Bouldin en fonction du nombre du nombre de composantes')
    ax[1].set_xlabel('Nombre de composantes')
    ax[1].set_ylabel('Score de Davies-Bouldin')
    plt.show()

    # ax[1].plot(K, ari)
    # ax[1].set_title('Coefficient de rand ajusté en fonction du nombre de composantes')
    # ax[1].set_xlabel('Nombre de composantes')
    # ax[1].set_ylabel('Coefficient de rand')
    # plt.show()

    # plt.plot(K, T)
    # plt.title("Temps d'éxécution de l'algorithme en fonction du nombre de composantes")
    # plt.xlabel('Nombre de composantes')
    # plt.ylabel("Temps d'éxécution")
    # plt.show()