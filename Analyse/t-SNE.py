import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import plotly.express as px
from  DBSCAN import DBSCAN_clust
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SpecClust import specClust
from CAH import CAH
from GMM import GMM
from kmeans import k_means

# 2.1 Examen des données 

dataset = pd.read_csv("./data/data.csv", index_col=0)
index=dataset.index

# 2.2 Pré-traitement des données

X=dataset.to_numpy()
# print(X)

scaler=StandardScaler(with_std=True)
Z=scaler.fit_transform(X)

# 2.3 Visualisation t-SNE
variable_to_color = 'total_fertility'

# Appliquer t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
Z_tsne = tsne.fit_transform(Z)

# Clustering 2D

# Données brutes corrigées centrées réduites avant le clustering
# clusters_DBSCAN=DBSCAN_clust(Z, 8) # Il n'existe pas de k faisant converger l'algorithme de manière pertinente
# clusters_spec=specClust(Z, 15, matrix='nearest_neighbors', KNN=4) # KNN=4 et k=15, 7, 11
# clusters_asc_hier=CAH(Z, 'ward', 21, 'maxclust') # Plus k est bas, mieux c'est ... ou 21, 19
# clusters_cah=CAH(Z, 'ward', 250, 'distance')
# clusters_GMM=GMM(Z, 21, 'full', 10, 100)[0] # k = 21, 14, 10, 8
# clusters_kmeans=k_means(Z, 21, "k-means++", 10, 'lloyd', 100)[1] # k=21, 19, 10, 8

# Données passées par t-SNE avant le clustering
# clusters_DBSCAN=DBSCAN_clust(Z_tsne, 3) # Le meilleur k est 3 après avoir essayé toutes les possibilités
# clusters_spec=specClust(Z_tsne, 13, matrix='nearest_neighbors', KNN=4) # KNN=4 et k=13, 7, 17
# clusters_asc_hier=CAH(Z_tsne, 'ward', 10, 'maxclust') # Plus k est bas, mieux c'est ... 
# clusters_cah=clusters_asc_hier=CAH(Z_tsne, 'ward', 250, 'distance')
# clusters_GMM=GMM(Z_tsne, 13, 'full', 10, 100)[0] # k = 13
# clusters_kmeans=k_means(Z_tsne, 11, "k-means++", 10, 'lloyd', 100)[1] # k = 11

# Boncle de test des variables
# for k in range (2, 25):
#     print (f'\n Pour k={k}, voici les scores :')
#     k_means(Z, k, "k-means++", 10, 'lloyd', 100)[1]

# Créer un DataFrame pour la visualisation avec Plotly
tsne_df = pd.DataFrame(Z_tsne, columns=['TSNE1', 'TSNE2'])
tsne_df['Country'] = index
tsne_df['Variable'] = scaler.fit_transform(dataset[[variable_to_color]])  # Ajouter la variable pour la couleur

# Visualisation interactive avec Plotly
fig = px.scatter(
    tsne_df, 
    x='TSNE1', 
    y='TSNE2', 
    hover_name='Country',  # Noms des pays affichés au survol
    title='Visualisation t-SNE du jeu de données en 2D',
    labels={'TSNE1': 'Dimension 1', 'TSNE2': 'Dimension 2', 'Variable': variable_to_color},  # Étiquettes des axes
    width=800, height=600,
    color='Variable',  # Utiliser la variable pour la coloration
    color_continuous_scale='turbo'  # Utilisez 'viridis' ou autre colormap si nécessaire
)

# Afficher le graphique interactif
fig.show()

# Fonction d'affichage des clusters en 2 dimensions
def plot_clusters_2d(data_2d, clusters, labels, title="Visualisation t-SNE 2D des clusters"):

    tsne_df = pd.DataFrame(data_2d, columns=['TSNE1', 'TSNE2'])
    tsne_df['Cluster'] = clusters.astype(str)  # Convertir en chaîne pour la couleur catégorielle
    tsne_df['Country'] = labels

    # Graphique 2D avec clusters
    fig_2d = px.scatter(
        tsne_df, 
        x='TSNE1', 
        y='TSNE2',
        hover_name='Country',
        color='Cluster',
        title=title,
        labels={'TSNE1': 'Dimension 1', 'TSNE2': 'Dimension 2'},
        width=800, height=800
    )

    # Légende
    clusters_dict = tsne_df.groupby('Cluster')['Country'].apply(list).to_dict()
    table_data = []
    for cluster, countries in clusters_dict.items():
        table_data.append(
            [f"Cluster {cluster} " + "\n" + f"Effectif={len(countries)}", ", ".join(countries)]
        )

    # Créer une sous-figure avec le graphe 2D et la légende
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        specs=[[{"type": "scatter"}, {"type": "table"}]]
    )
    for trace in fig_2d.data:
        fig.add_trace(trace, row=1, col=1)

    # Ajouter le tableau avec les clusters et les pays
    fig.add_trace(
        go.Table(
            header=dict(values=["Cluster", "Pays"], align='left', font=dict(size=12)),
            cells=dict(values=list(zip(*table_data)), align='left')
        ),
        row=1, col=2
    )

    # Afficher le résultat
    fig.update_layout(
        title=title,
        showlegend=False,
        width=1200,
        height=800
    )
    fig.show()

# plot_clusters_2d(Z_tsne, clusters=clusters_kmeans, labels=index)
# plot_clusters_2d(Z_tsne, clusters=clusters_cah, labels=index)
# plot_clusters_2d(Z_tsne, clusters=clusters_asc_hier, labels=index)
# plot_clusters_2d(Z_tsne, clusters=clusters_GMM, labels=index)
# plot_clusters_2d(Z_tsne, clusters=clusters_DBSCAN, labels=index)
# plot_clusters_2d(Z_tsne, clusters=clusters_spec, labels=index)



# Appliquer t-SNE en 3D
tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
Z_tsne = tsne.fit_transform(Z)

# # Clustering 3D

# Données brutes centrées réduites corrigées avant le clustering
# clusters_DBSCAN=DBSCAN_clust(Z, 8) # Il n'existe pas de k faisant converger l'algorithme de manière pertinente
# clusters_spec=specClust(Z, 15, matrix='nearest_neighbors', KNN=4) # KNN=4 et k=15, 7, 11
# clusters_asc_hier=CAH(Z, 'ward', 21, 'maxclust') # Plus k est bas, mieux c'est ... ou 21, 19
# clusters_cah=CAH(Z, 'ward', 250, 'distance')
# clusters_GMM=GMM(Z, 21, 'full', 10, 100)[0] # k = 21, 14, 10, 8
# clusters_kmeans=k_means(Z, 21, "k-means++", 10, 'lloyd', 100)[1] # k=21, 19, 10, 8

# Données passées par t-SNE avant le clustering
# clusters_DBSCAN=DBSCAN_clust(Z_tsne, 2) # Le meilleur k est 2 après avoir essayé toutes les possibilités
# clusters_spec=specClust(Z_tsne, 8, matrix='nearest_neighbors', KNN=4) # KNN dans [3, 4, 5, 6, 7] avec max à 4 puis 6 et k=3, 8, 7, 4, 5, 6
# clusters_asc_hier=CAH(Z_tsne, 'ward', 11, 'maxclust') # Trouver le bon seuil t=3,4
# clusters_cah=clusters_asc_hier=CAH(Z_tsne, 'ward', 250, 'distance')
# clusters_GMM=GMM(Z_tsne, 11, 'full', 10, 100)[0] # k = 6, 3, 11, 5
# clusters_kmeans=k_means(Z_tsne, 11, "k-means++", 10, 'lloyd', 100)[1] # k=3, 4, 5, 6, 11, 8

# Boncle de test des variables
# for k in range (2, 12):
#     print (f'\n Pour k={k}, voici les scores :')
#     clusters_kmeans=k_means(Z_tsne, k, "k-means++", 10, 'lloyd', 100)[1]

# Créer un DataFrame pour la visualisation avec Plotly
tsne_df = pd.DataFrame(Z_tsne, columns=['TSNE1', 'TSNE2', 'TSNE3'])
tsne_df['Country'] = index  # Ajouter les noms des pays dans la colonne 'Country'
tsne_df['Variable'] = scaler.fit_transform(dataset[[variable_to_color]])  # Ajouter la variable pour la couleur

# Visualisation interactive 3D avec Plotly
fig = px.scatter_3d(
    tsne_df, 
    x='TSNE1', 
    y='TSNE2', 
    z='TSNE3',  # Troisième dimension pour le 3D
    hover_name='Country',  # Noms des pays affichés au survol
    title='Visualisation t-SNE 3D du jeu de données',
    labels={'TSNE1': 'Dimension 1', 'TSNE2': 'Dimension 2', 'TSNE3': 'Dimension 3', 'Variable': variable_to_color},  # Étiquettes des axes
    width=800, height=800,
    color='Variable',  # Utiliser la variable pour la coloration
    color_continuous_scale='turbo'  # Utilisez 'viridis' ou autre colormap si nécessaire
)
fig.show()

# Fonction d'affichage des clusters en 2 dimensions
def plot_clusters_3d(data_3d, clusters, labels, title="Visualisation t-SNE 3D des clusters"):

    # Création d'un DataFrame pour la visualisation
    tsne_df = pd.DataFrame(data_3d, columns=['TSNE1', 'TSNE2', 'TSNE3'])
    tsne_df['Cluster'] = clusters.astype(str)  # Convertir en chaîne pour la couleur catégorielle
    tsne_df['Country'] = labels

    # Graphique 3D avec clusters
    fig_3d = px.scatter_3d(
        tsne_df, 
        x='TSNE1', 
        y='TSNE2', 
        z='TSNE3',
        hover_name='Country',
        color='Cluster',
        title=title,
        labels={'TSNE1': 'Dimension 1', 'TSNE2': 'Dimension 2', 'TSNE3': 'Dimension 3'},
        width=800, height=800
    )

    # Légende
    clusters_dict = tsne_df.groupby('Cluster')['Country'].apply(list).to_dict()
    table_data = []
    for cluster, countries in clusters_dict.items():
        table_data.append(
            [f"Cluster {cluster} "+"\n"+f"Effectif={len(countries)}", ", ".join(countries)]
        )

    # Créer une sous-figure avec le graphe 3D et la légende
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        specs=[[{"type": "scatter3d"}, {"type": "table"}]]
    )
    for trace in fig_3d.data:
        fig.add_trace(trace, row=1, col=1)

    # Ajouter la table avec les clusters et pays
    fig.add_trace(
        go.Table(
            header=dict(values=["Cluster", "Pays"], align='left', font=dict(size=12)),
            cells=dict(values=list(zip(*table_data)), align='left')
        ),
        row=1, col=2
    )

    # Afficher le résultat
    fig.update_layout(
        title=title,
        showlegend=False,
        width=1200,
        height=800
    ) 
    fig.show()

# plot_clusters_3d(Z_tsne, clusters=clusters_kmeans, labels=index)
# plot_clusters_3d(Z_tsne, clusters=clusters_cah, labels=index)
# plot_clusters_3d(Z_tsne, clusters=clusters_asc_hier, labels=index)
# plot_clusters_3d(Z_tsne, clusters=clusters_GMM, labels=index)
# plot_clusters_3d(Z_tsne, clusters=clusters_DBSCAN, labels=index)
# plot_clusters_3d(Z_tsne, clusters=clusters_spec, labels=index)
