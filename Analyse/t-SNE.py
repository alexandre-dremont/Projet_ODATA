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

# 2.1 Examen des données 

dataset = pd.read_csv("./data/data.csv", index_col=0)
# print(dataset)
# print(dataset.info())
# print(dataset.describe())
# print(dataset.shape)
# print(dataset.head())
# print(dataset.columns)
index=dataset.index

# Que des données de type float sauf le salaire qui est entier et le nom des pays un object (string en l'occurence)

# print(dataset.isna())
# print(dataset.isnull())

# print(dataset["total_fertility"]=="NA")

# Nombre total de valeurs manquantes pour chaque caractéristique
# print(dataset.isnull().sum()) # Il manque deux valeurs dans les colonnes "total_fertility" et "GDP"
# print(dataset.isna().sum())

# Le taux de fertilité n'a pas été renseigné en France et au Niger et le PIB en Italie et en Norvège

# print(dataset.isna().sum(axis=1))
# Nombre total de valeurs manquantes
# print(dataset.isnull().sum().sum())


# dataset.hist()
# plt.show()

# print(dataset[dataset["child_mortality"]>130])
# Corriger le taux de mortalité infantile pour Haïti (57 pour 1000), Sierra Leone (101), Tchad (103)

# print(dataset[dataset["exports"]<5])
# Myanmar

# print(dataset[dataset["health"]>10])
# USA à 17,9/1000

# print(dataset[dataset["imports"]<15])
# Myanmar à nouveau, le Japon et le Brésil importent peu

# print(dataset[dataset["income"]>75000])
# Salaire moyen au Qatar aberrant (125000 contre )

# print(dataset[dataset["inflation"]>25])
# Inflation moyen au Nigeria aberrante (104.0)

# print(dataset[dataset["life_expectation"]<50])
# On corrige les espérances de vie du Bangladesh (0 contre réellement 62.5) et d'Haïti (32.1 contre réellement 63.2) par pro-rata

# print(dataset[dataset["GDP"]>75000])
# PIB par habitant de l'Australie, de la Grande Bretagne et des Etats-Unis à corriger

# salaire et PIB par habitant corrélés positivement ?
# imports et exports corrélés positivement ?
# mortalité infantile et santé ou espérance de vie corrélés négativement ?

# 2.2 Pré-traitement des données

X=dataset.to_numpy()
# print(X)

scaler=StandardScaler(with_std=True)
Z=scaler.fit_transform(X)

# print(np.mean(Z))
# print(np.var(Z))
# print(Z)

# print(dataset.corr())

# pd.plotting.scatter_matrix(frame=dataset)
# plt.show()

# 2.3 Visualisation t-SNE

variable_to_color = 'life_expectation'

# # Appliquer t-SNE
# tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
# Z_tsne = tsne.fit_transform(Z)

# # Créer un DataFrame pour la visualisation avec Plotly
# tsne_df = pd.DataFrame(Z_tsne, columns=['TSNE1', 'TSNE2'])
# tsne_df['Country'] = index
# tsne_df['Variable'] = scaler.fit_transform(dataset[[variable_to_color]])  # Ajouter la variable pour la couleur

# # Visualisation interactive avec Plotly
# fig = px.scatter(
#     tsne_df, 
#     x='TSNE1', 
#     y='TSNE2', 
#     hover_name='Country',  # Noms des pays affichés au survol
#     title='Visualisation t-SNE du jeu de données',
#     labels={'TSNE1': 'Dimension 1', 'TSNE2': 'Dimension 2', 'Variable': variable_to_color},  # Étiquettes des axes
#     width=800, height=600,
#     color='Variable',  # Utiliser la variable pour la coloration
#     color_continuous_scale='turbo'  # Utilisez 'viridis' ou autre colormap si nécessaire
# )

# # Afficher le graphique interactif
# fig.show()

# Appliquer t-SNE en 3D
tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
Z_tsne = tsne.fit_transform(Z)

# clusters_DBSCAN=DBSCAN_clust(Z_tsne, 2) # Le meilleur k est 2 après avoir essayé toutes les possibilités
# clusters_spec=specClust(Z_tsne, 8, matrix='nearest_neighbors', KNN=4) # KNN dans [3, 4, 5, 6, 7] avec max à 4 puis 6 et k=3, 8, 7, 4, 5, 6
# clusters_asc_hier=CAH(Z_tsne, 'ward', 4, 'maxclust') # Trouver le bon seuil t=2,3,4
clusters_cah=clusters_asc_hier=CAH(Z_tsne, 'ward', 250, 'distance')

# for k in range (2700, 2800, 5):
#     print (f'\n Pour k={k}, voici les scores :')

# # Créer un DataFrame pour la visualisation avec Plotly
# tsne_df = pd.DataFrame(Z_tsne, columns=['TSNE1', 'TSNE2', 'TSNE3'])
# tsne_df['Country'] = index  # Ajouter les noms des pays dans la colonne 'Country'
# tsne_df['Variable'] = scaler.fit_transform(dataset[[variable_to_color]])  # Ajouter la variable pour la couleur

# # Visualisation interactive 3D avec Plotly
# fig = px.scatter_3d(
#     tsne_df, 
#     x='TSNE1', 
#     y='TSNE2', 
#     z='TSNE3',  # Troisième dimension pour le 3D
#     hover_name='Country',  # Noms des pays affichés au survol
#     title='Visualisation t-SNE 3D du jeu de données',
#     labels={'TSNE1': 'Dimension 1', 'TSNE2': 'Dimension 2', 'TSNE3': 'Dimension 3', 'Variable': variable_to_color},  # Étiquettes des axes
#     width=800, height=800,
#     color='Variable',  # Utiliser la variable pour la coloration
#     color_continuous_scale='turbo'  # Utilisez 'viridis' ou autre colormap si nécessaire
# )

# # Afficher le graphique interactif
# fig.show()

def plot_clusters_3d_with_legend(data_3d, clusters, labels, title="Visualisation t-SNE 3D des clusters"):
    """
    Affiche une visualisation 3D des données projetées et une légende mosaïque avec les pays groupés par clusters.

    Paramètres:
    -----------
    data_3d : array-like
        Tableau numpy ou DataFrame contenant les données projetées en 3D (e.g., résultats de t-SNE).
    clusters : list ou array
        Liste ou tableau des identifiants de cluster pour chaque point de données.
    labels : list
        Liste des noms des points (par exemple, des pays), pour afficher au survol.
    title : str, optional
        Titre du graphique.

    Retour:
    -------
    Affiche le graphique interactif 3D avec la légende des clusters.
    """
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

    # Créer un DataFrame groupé par cluster pour la légende
    clusters_dict = tsne_df.groupby('Cluster')['Country'].apply(list).to_dict()
    
    # Formatage de la légende pour chaque cluster
    table_data = []
    for cluster, countries in clusters_dict.items():
        table_data.append(
            [f"Cluster {cluster}", ", ".join(countries)]
        )

    # Créer une sous-figure avec le graphe 3D et la légende
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        specs=[[{"type": "scatter3d"}, {"type": "table"}]]
    )

    # Ajouter le graphique 3D
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

    # Afficher le graphique complet
    fig.update_layout(
        title=title,
        showlegend=False,
        width=1200,
        height=800
    )
    
    fig.show()

# plot_clusters_3d_with_legend(Z_tsne, clusters=clusters_DBSCAN, labels=index)
plot_clusters_3d_with_legend(Z_tsne, clusters=clusters_cah, labels=index)
