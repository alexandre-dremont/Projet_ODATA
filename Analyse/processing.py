import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans

# Récupération des données
# Chaque fichier contient 3 colonnes. Pour chaque individu (ou point), les 2
# premières colonnes correspondent aux valeurs de 2 caractéristiques, la 3ème
# colonne indique sa classe d’appartenance.

jain=pd.read_csv("./data/jain.txt", sep="\t")
# print(jain)
# print(jain.info())
# print(jain.describe())
# print(jain.shape)
# print(jain.head())
# print(jain.columns)

aggregation=pd.read_csv("./data/aggregation.txt", sep="\t")
# print(aggregation)
# print(aggregation.info())
# print(aggregation.describe())
# print(aggregation.shape)
# print(aggregation.head())
# print(aggregation.columns)
# print(aggregation.index)

pathbased=pd.read_csv("./data/pathbased.txt", sep="\t")
# print(pathbased)
# print(pathbased.info())
# print(pathbased.describe())
# print(pathbased.shape)
# print(pathbased.head())
# print(pathbased.columns)
# print(pathbased.index)


# 2.1 Examen des données 

dataset = pd.read_csv("./data/data.csv", index_col=0)
# print(dataset)
# print(dataset.info())
print(dataset.describe())
# print(dataset.shape)
print(dataset.head())
print(dataset.columns)
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
pca=PCA()
pca.fit(Z)

# print(np.mean(Z))
# print(np.var(Z))
# print(Z)

print(dataset.corr())

pd.plotting.scatter_matrix(frame=dataset)
plt.show()