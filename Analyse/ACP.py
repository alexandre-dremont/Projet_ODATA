import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Circle
from kmeans import k_means, choix_K_kmeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from GMM import GMM, choix_K_gmm
from CAH import CAH
from DBSCAN import DBSCAN_clust
from SpecClust import specClust

dataset = pd.read_csv("./data/data.csv", index_col=0)

col=dataset.columns

dataset.drop(["Singapore", "Malta", "Luxembourg"], axis=0, inplace=True) # Essayer de les projeter dans le nouvel espace
print(dataset.shape)
index_data=dataset.index
print(index_data.shape)

# 2.2 Pré-traitement des données

X=dataset.to_numpy()
# print(X)

scaler=StandardScaler()
Z=scaler.fit_transform(X)

# print(dataset.corr())
# Corrélation positive la plus forte entre le taux de fécondité et le taux de mortalité enfantile (0.84) puis entre les imports et les exports (0.73)
# Corrélation négative la plus forte entre l'espérance de vie et la mortalité infantile (-0.76) puis entre le taux de fécondité et l'espérance de vie (-0.63)

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
# print(explained_variance_ratio)
pourcentage_inertie=explained_variance_ratio/sum(explained_variance_ratio)*100

inertie_cumulee = np.cumsum(explained_variance_ratio)

proj = pca.transform(Z) # Effectue la projection de Z sur les axes principaux, le résultat obtenu est une matrice des coordonnées des individus dans l'espace des axes principaux
print(proj.shape)

def affichage():

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

    # proj = pca.transform(Z) # Effectue la projection de Z sur les axes principaux, le résultat obtenu est une matrice des coordonnées des individus dans l'espace des axes principaux

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

def plot_clusters_3d_with_legend(data_3d, clusters, labels, title="Visualisation ACP 3D des clusters"):
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
    tsne_df = pd.DataFrame(data_3d, columns=['Axe 1', 'Axe 2', 'Axe 3'])
    tsne_df['Cluster'] = clusters.astype(str)  # Convertir en chaîne pour la couleur catégorielle
    tsne_df['Country'] = labels

    # Graphique 3D avec clusters
    fig_3d = px.scatter_3d(
        tsne_df, 
        x='Axe 1', 
        y='Axe 2', 
        z='Axe 3',
        hover_name='Country',
        color='Cluster',
        title=title,
        labels={'Axe 1': 'Dimension 1', 'Axe 2': 'Dimension 2', 'Axe 3': 'Dimension 3'},
        width=800, height=800
    )

    # Créer un DataFrame groupé par cluster pour la légende
    clusters_dict = tsne_df.groupby('Cluster')['Country'].apply(list).to_dict()
    
    # Formatage de la légende pour chaque cluster
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

# ACP

# plot_clusters_3d_with_legend(proj[:,:3], np.zeros((index_data.size,1)),index_data)

# K-means

# _,labels_kmeans,__=k_means(proj[:,:3], 11, "k-means++", 10, 'lloyd', 100)
# choix_K_kmeans(proj, "k-means++", 10, 'lloyd', 100) # K = 7 ou 11

# plot_clusters_3d_with_legend(proj[:,:3], clusters=labels_kmeans, labels=index_data) K=11

# print(labels)
# print(proj[:,:3])

# CAH

# for t in range (70, 134, 2): 
#     print (f'\n Pour k={t}, voici les scores :')
#     labels_cah=CAH(proj[:,:3], "ward", t/10, "distance")

# labels_cah=CAH(proj[:,:3], "ward", 8.8, "distance") #k=8.8 ou 13.2
# plot_clusters_3d_with_legend(proj[:,:3], clusters=labels_cah, labels=index_data)

# GMM

# labels_gmm,_=GMM(proj[:,:3], 12, 'full', 10, 100)
# choix_K_gmm(proj[:,:3], 'full', 10, 100) # K = 8 ou 12 

# plot_clusters_3d_with_legend(proj[:,:3], clusters=labels_gmm, labels=index_data) # K = 8

# DBSCAN

# for k in range (2, 9):
#     print (f'\n Pour k={k}, voici les scores :')
#     labels_dbscan=DBSCAN_clust(proj, k)

# k=2

# labels_dbscan=DBSCAN_clust(proj[:,:3], 2) 
# print(labels_dbscan)
# plot_clusters_3d_with_legend(proj[:,:3], clusters=labels_dbscan, labels=index_data)

# Clustering Spectral

# for k in range (2, 20):
#     print (f'\n Pour k={k}, voici les scores :')
#     labels_dbscan=specClust(proj[:,:3], k, "rbf")

# K = 5 ou 7

# for k in range (2, 20):
#     print (f'\n Pour k={k}, voici les scores :')
#     specClust(proj[:,:3], k , matrix='nearest_neighbors', KNN=5)

# labels_cluspec=specClust(proj[:,:3], 16 , matrix='nearest_neighbors', KNN=5) # k=4, 16 ou 18
# plot_clusters_3d_with_legend(proj[:,:3], clusters=labels_cluspec, labels=index_data)

# K=7

# Calculer les contributions d'une dizaine de pays en besoin aux trois axes