from kmeans import k_means, choix_K_kmeans
from GMM import GMM, choix_K_gmm
from CAH import CAH
from DBSCAN import DBSCAN_clust
from SpecClust import specClust
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ACP import plot_clusters_3d_with_legend

dataset = pd.read_csv("./data/data.csv", index_col=0)

dataset.drop(["Singapore", "Malta", "Luxembourg"], axis=0, inplace=True)
index_data=dataset.index

X=dataset.to_numpy()
# print(X)

scaler=StandardScaler()
Z=scaler.fit_transform(X)

pca=PCA()
pca.fit(Z)

proj = pca.transform(Z)

# K-means

# _,labels_kmeans,__=k_means(Z, 21, "k-means++", 10, 'lloyd', 100)
# choix_K_kmeans(Z, "k-means++", 10, 'lloyd', 100) # K = 16 ou 17 , 21 puis 17

# plot_clusters_3d_with_legend(proj[:,:3], clusters=labels_kmeans, labels=index_data) # K=21

# CAH

# for t in range (70, 134, 2): 
#     print (f'\n Pour k={t}, voici les scores :')
#     labels_cah=CAH(Z, "ward", t/10, "distance")

# labels_cah=CAH(Z, "ward", 21, "maxclust")
# plot_clusters_3d_with_legend(proj[:,:3], clusters=labels_cah, labels=index_data)

# GMM

# labels_gmm,_= GMM(Z, 17, 'full', 10, 100)
# choix_K_gmm(Z, 'full', 10, 100) # K = 8, 15 ou 17

# plot_clusters_3d_with_legend(proj[:,:3], clusters=labels_gmm, labels=index_data) # K = 15

# DBSCAN

# for k in range (2, 9):
#     print (f'\n Pour k={k}, voici les scores :')
#     labels_dbscan=DBSCAN_clust(Z, k)

# labels_dbscan=DBSCAN_clust(Z, 5) 
# plot_clusters_3d_with_legend(proj[:,:3], clusters=labels_dbscan, labels=index_data)

# Clustering Spectral

# for k in range (2, 20):
#     print (f'\n Pour k={k}, voici les scores :')
#     labels_dbscan=specClust(Z, k, "rbf")

# for k in range (2, 20):
#     print (f'\n Pour k={k}, voici les scores :')
#     specClust(Z, k , matrix='nearest_neighbors', KNN=4)

# labels_cluspec=specClust(Z, 15 , matrix='nearest_neighbors', KNN=4)
# plot_clusters_3d_with_legend(proj[:,:3], clusters=labels_cluspec, labels=index_data)