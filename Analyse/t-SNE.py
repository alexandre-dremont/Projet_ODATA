import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import plotly.express as px

# 2.1 Examen des données 

dataset = pd.read_csv("./data/data.csv", index_col=0)
# print(dataset)
# print(dataset.info())
print(dataset.describe())
# print(dataset.shape)
print(dataset.head())
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

variable_to_color = 'total_fertility'

# Appliquer t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
Z_tsne = tsne.fit_transform(Z)

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
    title='Visualisation t-SNE du jeu de données',
    labels={'TSNE1': 'Dimension 1', 'TSNE2': 'Dimension 2', 'Variable': variable_to_color},  # Étiquettes des axes
    width=800, height=600,
    color='Variable',  # Utiliser la variable pour la coloration
    color_continuous_scale='viridis'  # Utilisez 'viridis' ou autre colormap si nécessaire
)

# Afficher le graphique interactif
fig.show()

# # Appliquer t-SNE en 3D
# tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
# Z_tsne = tsne.fit_transform(Z)

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
#     color_continuous_scale='viridis'  # Utilisez 'viridis' ou autre colormap si nécessaire
# )

# # Afficher le graphique interactif
# fig.show()