from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd

jain=pd.read_csv("./data/jain.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"])

aggregation=pd.read_csv("./data/aggregation.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"])

pathbased=pd.read_csv("./data/pathbased.txt", sep="\t", names=["1ère caractéristique", "2ème caractéristique", "Résultat"])

gm = GaussianMixture(n_components=2, ).fit(jain.drop(columns=["Résultat"]))
print(gm.means_)
print(gm.predict([[0, 0], [12, 3]]))