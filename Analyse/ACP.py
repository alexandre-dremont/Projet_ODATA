import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Circle

dataset = pd.read_csv("./data/data.csv", index_col=0)
index_data=dataset.index
col=dataset.columns

# 2.2 Pré-traitement des données

X=dataset.to_numpy()
# print(X)

scaler=StandardScaler()
Z=scaler.fit_transform(X)

# print(dataset.corr())
# Corrélation positive la plus forte entre le taux de natalité et le taux de mortalité enfantile (0.84) puis entre les imports et les exports (0.73)
# Corrélation négative la plus forte entre l'espérance de vie et la mortalité enfantile (-0.76) puis entre le taux de natalité et l'espérance de vie (-0.63)

# pd.plotting.scatter_matrix(frame=dataset)
# plt.show()


# 2.3 Visualisation des données

pca=PCA()
pca.fit(Z)

# print(np.mean(Z))
# print(np.var(Z))
# print(Z)

# Détermintaion du nombre d'axes à conserver

vecteurs_propres = pca.components_
valeurs_propres = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
print(explained_variance_ratio)
pourcentage_inertie=explained_variance_ratio/sum(explained_variance_ratio)*100

inertie_cumulee = np.cumsum(explained_variance_ratio)

# Créer une figure avec 2 sous-fenêtres côte à côte
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Première fenêtre : diagramme en barres des inerties (variance expliquée)
ax[0].bar(np.arange(1, len(explained_variance_ratio) + 1), explained_variance_ratio, color='lightgreen', edgecolor='grey')
ax[0].set_title("Variance expliquée par chaque composante")
ax[0].set_xlabel("Composante principale")
ax[0].set_ylabel("Variance expliquée (%)")

# Deuxième fenêtre : courbe de l'inertie cumulée
ax[1].plot(np.arange(1, len(inertie_cumulee) + 1), inertie_cumulee, marker='o', linestyle='-', color='b')
ax[1].bar(np.arange(1, len(inertie_cumulee) + 1), inertie_cumulee, color='lightblue', edgecolor='grey')
ax[1].set_title("Inertie cumulée")
ax[1].set_xlabel("Nombre de composantes principales")
ax[1].set_ylabel("Inertie cumulée (%)")
ax[1].grid()
plt.show()

# On garde 3 axes


# Projection sur les axes principaux

proj = pca.transform(Z) # Effectue la projection de Z sur les axes principaux, le résultat obtenu est une matrice des coordonnées des individus dans l'espace des axes principaux

fig, ax = plt.subplots(2, 3, figsize=(13, 5))

# Affichage des villes dans le plan axe1 - axe2 (faire éventuellement de même avec le plan axe3 - axe4 voire en 3 dimensions)
ax[0][0].scatter(proj[:, 0], proj[:, 1], color='blue')
for i, txt in enumerate(index_data):
    ax[0][0].text(proj[i, 0], proj[i, 1], txt, fontsize=9)
ax[0][0].set_title('Projection des individus sur le plan principal 1-2')
ax[0][0].set_xlabel('Axe 1')
ax[0][0].set_ylabel('Axe 2')
ax[0][0].grid(True)

# Affichage des villes dans le plan axe1 - axe3 (faire éventuellement de même avec le plan axe3 - axe4 voire en 3 dimensions)
ax[0][1].scatter(proj[:, 0], proj[:, 2], color='blue')
for i, txt in enumerate(index_data):
    ax[0][1].text(proj[i, 0], proj[i, 2], txt, fontsize=9)
ax[0][1].set_title('Projection des individus sur le plan principal 1-3')
ax[0][1].set_xlabel('Axe 1')
ax[0][1].set_ylabel('Axe 3')
ax[0][1].grid(True)

# Affichage des villes dans le plan axe2 - axe3 (faire éventuellement de même avec le plan axe3 - axe4 voire en 3 dimensions)
ax[0][2].scatter(proj[:, 1], proj[:, 2], color='blue')
for i, txt in enumerate(index_data):
    ax[0][2].text(proj[i, 1], proj[i, 2], txt, fontsize=9)
ax[0][2].set_title('Projection des individus sur le plan principal 2-3')
ax[0][2].set_xlabel('Axe 2')
ax[0][2].set_ylabel('Axe 3')
ax[0][2].grid(True)

# Contributions des individus pour chaque axe
contributions = (proj ** 2) / valeurs_propres

# Contributions totales par axe (somme pour chaque individu)
contributions_totales = contributions.sum(axis=0)

# Trouver les indices des individus qui contribuent le plus (en positif et négatif)
for k in range(3): #for k in range(len(valeurs_propres)):
    # Contributions maximales et minimales pour chaque axe k
    ind_max = np.argmax(contributions[:, k])
    ind_min = np.argmin(contributions[:, k])
    print(f"Pour l'axe {k+1}, contribution maximale : {index_data[ind_max]}, contribution minimale : {index_data[ind_min]}")

# Calcul des corrélations entre les variables et les axes principaux
corr_variables_factors = vecteurs_propres.T * np.sqrt(valeurs_propres)

# Afficher les corrélations entre les variables et les deux premiers axes (1 et 2)
print("Corrélations variables - Axe 1 et Axe 2 :")
print(corr_variables_factors[:, :2])

# Tracer le cercle des corrélations
cercle = Circle((0, 0), 1, facecolor='none', edgecolor='b', linestyle='--')
plt.gca().add_patch(cercle)

# Limites du graphique (rayon = 1)
ax[1][0].set_xlim(-1.1, 1.1)
ax[1][0].set_ylim(-1.1, 1.1)

# Tracer les variables projetées dans le plan 1-2
for i in range(len(corr_variables_factors)):
    # Flèche pour chaque variable
    ax[1][0].arrow(0, 0, corr_variables_factors[i, 0], corr_variables_factors[i, 1], 
              head_width=0.05, head_length=0.05, color='r')
    # Annoter chaque variable
    ax[1][0].text(corr_variables_factors[i, 0], corr_variables_factors[i, 1], col[i], fontsize=12)  # col contient les noms des colonnes

ax[1][0].set_xlabel('Axe 1')
ax[1][0].set_ylabel('Axe 2')
ax[1][0].set_title('Cercle des corrélations - Plan 1-2')
ax[1][0].grid(True)
ax[1][0].axhline(0, color='black',linewidth=0.5)
ax[1][0].axvline(0, color='black',linewidth=0.5)

# Afficher les corrélations entre les variables et les deux premiers axes (1 et 3)
print("Corrélations variables - Axe 1 et Axe 3 :")
print(corr_variables_factors[:, :3])

# Tracer le cercle des corrélations
cercle = Circle((0, 0), 1, facecolor='none', edgecolor='b', linestyle='--')
plt.gca().add_patch(cercle)

# Limites du graphique (rayon = 1)
ax[1][1].set_xlim(-1.1, 1.1)
ax[1][1].set_ylim(-1.1, 1.1)

# Tracer les variables projetées dans le plan 1-3
for i in range(len(corr_variables_factors)):
    # Flèche pour chaque variable
    ax[1][1].arrow(0, 0, corr_variables_factors[i, 0], corr_variables_factors[i, 2], 
              head_width=0.05, head_length=0.05, color='r')
    # Annoter chaque variable
    ax[1][1].text(corr_variables_factors[i, 0], corr_variables_factors[i, 2], col[i], fontsize=12)  # col contient les noms des colonnes

ax[1][1].set_xlabel('Axe 1')
ax[1][1].set_ylabel('Axe 3')
ax[1][1].set_title('Cercle des corrélations - Plan 1-3')
ax[1][1].grid(True)
ax[1][1].axhline(0, color='black',linewidth=0.5)
ax[1][1].axvline(0, color='black',linewidth=0.5)

# Afficher les corrélations entre les variables et les deux premiers axes (2 et 3)
print("Corrélations variables - Axe 1 et Axe 2 :")
print(corr_variables_factors[:, :2])

# Tracer le cercle des corrélations
cercle = Circle((0, 0), 1, facecolor='none', edgecolor='b', linestyle='--')
plt.gca().add_patch(cercle)

# Limites du graphique (rayon = 1)
ax[1][2].set_xlim(-1.1, 1.1)
ax[1][2].set_ylim(-1.1, 1.1)

# Tracer les variables projetées dans le plan 2-3
for i in range(len(corr_variables_factors)):
    # Flèche pour chaque variable
    ax[1][2].arrow(0, 0, corr_variables_factors[i, 1], corr_variables_factors[i, 2], 
              head_width=0.05, head_length=0.05, color='r')
    # Annoter chaque variable
    ax[1][2].text(corr_variables_factors[i, 1], corr_variables_factors[i, 2], col[i], fontsize=12)  # col contient les noms des colonnes

ax[1][2].set_xlabel('Axe 2')
ax[1][2].set_ylabel('Axe 3')
ax[1][2].set_title('Cercle des corrélations - Plan 2-3')
ax[1][2].grid(True)
ax[1][2].axhline(0, color='black',linewidth=0.5)
ax[1][2].axvline(0, color='black',linewidth=0.5)

plt.tight_layout()
plt.show()



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], color='blue')
for i, txt in enumerate(index_data):
    ax.text(proj[i, 0], proj[i, 1], proj[i, 2], txt, fontsize=9)
ax.set_title('Projection des individus dans l\'espace principal 1-2-3')
ax.set_xlabel('Axe 1')
ax.set_ylabel('Axe 2')
ax.set_zlabel('Axe 3')
ax.grid(True)
plt.show()

# Créer une figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Tracer la sphère des corrélations
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='b', alpha=0.1, edgecolor='none')

# Limites du graphique (rayon = 1)
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_zlim([-1.1, 1.1])

# Tracer les flèches 3D pour les variables projetées dans le plan 1-2-3
for i in range(len(corr_variables_factors)):
    ax.quiver(0, 0, 0, 
              corr_variables_factors[i, 0], corr_variables_factors[i, 1], corr_variables_factors[i, 2],
              color='r', arrow_length_ratio=0.1)  # Flèche 3D
    
    # Annoter chaque variable
    ax.text(corr_variables_factors[i, 0], corr_variables_factors[i, 1], corr_variables_factors[i, 2], 
            col[i], fontsize=12)

# Configuration des axes
ax.set_xlabel('Axe 1')
ax.set_ylabel('Axe 2')
ax.set_zlabel('Axe 3')
ax.set_title('Sphère des corrélations - Plan 1-2-3')

# Affichage de la grille
ax.grid(True)

# Afficher la figure
plt.show()