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

agg=pd.read_csv("./data/aggregation.txt", sep="\t")
# print(agg)
# print(agg.info())
# print(agg.describe())
# print(agg.shape)
# print(agg.head())
# print(agg.columns)
# print(agg.index)

pathbased=pd.read_csv("./data/pathbased.txt", sep="\t")
# print(pathbased)
# print(pathbased.info())
# print(pathbased.describe())
# print(pathbased.shape)
# print(pathbased.head())
# print(pathbased.columns)
# print(pathbased.index)