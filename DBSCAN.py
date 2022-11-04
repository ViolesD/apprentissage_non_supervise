#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:49:23 2022

@author: violes
"""


import numpy as np
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

import time
from sklearn import cluster
from sklearn import metrics
from sklearn import neighbors

from scipy.io import arff

path = './artificial/'
databrut = arff.loadarff(open(path+"xclara.arff",'r'))
databrut = arff.loadarff(open(path + "square1.arff", 'r'))
databrut = arff.loadarff(open(path+"sizes1.arff",'r'))
databrut = arff.loadarff(open(path+"simplex.arff",'r'))
databrut = arff.loadarff(open(path+"smile1.arff",'r'))
databrut = arff.loadarff(open(path+"smile3.arff",'r'))
databrut = arff.loadarff(open(path+"banana.arff",'r'))
#databrut = arff.loadarff(open(path+"complex9.arff",'r'))


datanp = [[x[0], x[1]] for x in databrut[0]]

f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

plt.scatter(f0, f1, s=8)
plt.title('Donnees initiales')
plt.show()

# Distances k plus proches voisins
# Donnees dans X

silhouette = []
davies = []
calinski = []

for k in range(2, 10):
    neigh = neighbors.NearestNeighbors(n_neighbors=k)
    neigh.fit(datanp)
    distances, indices = neigh.kneighbors(datanp)
    # retirer le point " origine "
    newDistances = np.asarray([np.average(distances[i][1:]) for i in range(0, distances.shape[0])])
    mean = newDistances.max()

    model = cluster.DBSCAN(min_samples=k, eps=mean)
    solution = model.fit(datanp)

    silhouette.append(metrics.silhouette_score(datanp, solution.labels_))
    davies.append(metrics.davies_bouldin_score(datanp, solution.labels_))
    calinski.append(metrics.calinski_harabasz_score(datanp, solution.labels_))

if (np.argmax(silhouette) == np.argmin(davies) and np.argmin(davies) == np.argmax(calinski)):
    k = np.argmax(calinski) + 2
    print("On a trouvé un gagnant")
else:
    print("Pas trouvé")
    k = 5


neigh = neighbors.NearestNeighbors(n_neighbors=k)
neigh.fit(datanp)
distances, indices = neigh.kneighbors(datanp)
# retirer le point " origine "
newDistances = np.asarray([np.average(distances[i][1:]) for i in range(0, distances.shape[0])])
mean = newDistances.max()
model = cluster.DBSCAN(min_samples=k, eps=mean)
solution = model.fit(datanp)

labels = model.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


plt.scatter(f0, f1, c=labels, s=8)
plt.title(" Resultat du clustering, nbr clusters: " + str(n_clusters_))
plt.show()

tpq2 = time.time()


# for k in range(1,10):
# model = cluster.AgglomerativeClustering( distance_threshold = k/20 , linkage = 'single' , n_clusters = None )
# model = model.fit( datanp )
# tps2 = time.time ()
# labels = model.labels_
# k = model.n_clusters_
# leaves = model.n_leaves_

#     if(k!=1):
#         silhouette.append(metrics.silhouette_score(datanp,labels))
#         davies.append(metrics.davies_bouldin_score(datanp,model.labels_))
#         calinski.append(metrics.calinski_harabasz_score(datanp,model.labels_))


# if(np.argmax(silhouette) == np.argmin(davies) and np.argmin(davies) == np.argmax(calinski) ):
#     k = np.argmax(calinski) +2
#     print("On a trouvé un gagnant : " + str(k)+ " clusters")

#     model = cluster.AgglomerativeClustering( distance_threshold = k/30 , linkage = 'single' , n_clusters = None )
#     model = model.fit( datanp )
#     labels = model.labels_


