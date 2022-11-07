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

import hdbscan
import pandas as pd

path = './artificial/'
#databrut = arff.loadarff(open(path+"xclara.arff",'r'))
#databrut = arff.loadarff(open(path + "square1.arff", 'r'))
#databrut = arff.loadarff(open(path+"sizes1.arff",'r'))
#databrut = arff.loadarff(open(path+"simplex.arff",'r'))
#databrut = arff.loadarff(open(path+"smile1.arff",'r'))
#databrut = arff.loadarff(open(path+"smile3.arff",'r'))
#databrut = arff.loadarff(open(path+"banana.arff",'r'))
#databrut = arff.loadarff(open(path+"complex9.arff",'r'))

#datanp = [[x[0], x[1]] for x in databrut[0]]

path2 = './dataset-rapport/'
databrut = pd.read_csv(path2+"x2.txt",sep=" ", encoding="ISO-8859-1", skipinitialspace=True)



datanp = databrut.to_numpy()

f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

plt.scatter(f0, f1, s=8)
plt.title('Donnees initiales')
plt.show()

# Distances k plus proches voisins
# Donnees dans X

tps1 = time.time()

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

    labels = solution.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    if (n_clusters_!=1):
        silhouette.append(metrics.silhouette_score(datanp, solution.labels_))
        davies.append(metrics.davies_bouldin_score(datanp, solution.labels_))
        calinski.append(metrics.calinski_harabasz_score(datanp, solution.labels_))
    else:
        silhouette.append(-99999999)
        davies.append(99999999)
        calinski.append(-99999999)

if (np.argmax(silhouette) == np.argmin(davies) and np.argmin(davies) == np.argmax(calinski)):
    k = np.argmax(calinski) + 2
    print("On a trouvé un gagnant : k=" + str(k))
elif (np.argmin(davies) == np.argmax(calinski)):
    k = np.argmax(calinski) + 2
    print("On a trouvé un gagnant : k=" + str(k))
elif (np.argmin(davies) == np.argmax(silhouette)):
    k = np.argmax(silhouette) + 2
    print("On a trouvé un gagnant : k=" + str(k))   
elif (np.argmax(silhouette) == np.argmax(calinski)):
    k = np.argmax(calinski) + 2
    print("On a trouvé un gagnant : k=" + str(k))
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

labels = solution.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
tps2 = time.time()


plt.scatter(f0, f1, c=labels, s=8)
plt.title(" Resultat du clustering DBSCAN, nbr clusters: " + str(n_clusters_))
plt.show()

print("nb clusters : " + str(n_clusters_) ," ,...  runtime = " , round ( ( tps2 - tps1 ) * 1000 , 2 ) , " ms" )



# tps1 = time.time()

# silhouette = []
# davies = []
# calinski = []
# #HDBSCAN method
# for k in range(2, 20):
#     model = hdbscan.HDBSCAN(min_cluster_size=k)
#     solution = model.fit(datanp)

#     silhouette.append(metrics.silhouette_score(datanp, solution.labels_))
#     davies.append(metrics.davies_bouldin_score(datanp, solution.labels_))
#     calinski.append(metrics.calinski_harabasz_score(datanp, solution.labels_))

# if (np.argmax(silhouette) == np.argmin(davies) and np.argmin(davies) == np.argmax(calinski)):
#     k = np.argmax(calinski) + 2
#     print("On a trouvé un gagnant : k=" + str(k))
# elif (np.argmin(davies) == np.argmax(calinski)):
#     k = np.argmax(calinski) + 2
#     print("On a trouvé un gagnant : k=" + str(k))
# elif (np.argmin(davies) == np.argmax(silhouette)):
#     k = np.argmax(silhouette) + 2
#     print("On a trouvé un gagnant : k=" + str(k))   
# elif (np.argmax(silhouette) == np.argmax(calinski)):
#     k = np.argmax(calinski) + 2
#     print("On a trouvé un gagnant : k=" + str(k))
# else:
#     print("Pas trouvé")
#     k = np.argmax(silhouette)+2

# print(np.argmax(silhouette))
# print(np.argmin(davies))
# print(np.argmax(calinski))
    
# model = hdbscan.HDBSCAN(min_cluster_size=k)
# solution = model.fit(datanp)

# labels = solution.labels_
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# tps2 = time.time()

# plt.scatter(f0, f1, c=labels, s=8)
# plt.title(" Resultat du clustering HDBSCAN, nbr clusters: " + str(n_clusters_))
# plt.show()
# print("nb clusters : " + str(n_clusters_) ," ,...  runtime = " , round ( ( tps2 - tps1 ) * 1000 , 2 ) , " ms" )
