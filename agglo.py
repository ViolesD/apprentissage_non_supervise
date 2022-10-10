#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:22:38 2022

@author: violes
"""


import numpy as np
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

import time
from sklearn import cluster
from sklearn import metrics


from scipy.io import arff

path = './artificial/'
# databrut = arff.loadarff(open(path+"xclara.arff",'r'))
databrut = arff.loadarff(open(path+"square1.arff",'r'))
# databrut = arff.loadarff(open(path+"sizes1.arff",'r'))
# databrut = arff.loadarff(open(path+"simplex.arff",'r'))
# databrut = arff.loadarff(open(path+"smile1.arff",'r'))
# databrut = arff.loadarff(open(path+"smile3.arff",'r'))
# databrut = arff.loadarff(open(path+"banana.arff",'r'))
# databrut = arff.loadarff(open(path+"complex9.arff",'r'))


datanp = [[x[0],x[1]] for x in databrut[0]]

f0= [f[0] for f in datanp]
f1= [f[1] for f in datanp]

plt.scatter(f0,f1,s=8)
plt.title('Donnees initiales')
plt.show()


# # Donnees dans datanp
# print ( " Dendrogramme 'single' donnees initiales " )
# linked_mat = shc.linkage ( datanp , 'single')
# plt.figure ( figsize = ( 12 , 12 ) )
# shc.dendrogram ( linked_mat ,
# orientation = 'top',
# distance_sort = 'descending', 
# show_leaf_counts = False )
# plt.show ()


silhouette =[]
davies = []
calinski = []
distmatrix = []


# set di stance_threshold ( 0 ensures we compute the full tree )
tps1 = time.time ()
for k in range(1,10):
    model = cluster.AgglomerativeClustering( distance_threshold = k/20 , linkage = 'single' , n_clusters = None )
    model = model.fit( datanp )
    tps2 = time.time ()
    labels = model.labels_
    k = model.n_clusters_
    leaves = model.n_leaves_
    
    if(k!=1):
        silhouette.append(metrics.silhouette_score(datanp,labels))
        davies.append(metrics.davies_bouldin_score(datanp,model.labels_))
        calinski.append(metrics.calinski_harabasz_score(datanp,model.labels_))
    
   
if(np.argmax(silhouette) == np.argmin(davies) and np.argmin(davies) == np.argmax(calinski) ):
    k = np.argmax(calinski) +2
    print("On a trouvé un gagnant : " + str(k)+ " clusters")
    
    model = cluster.AgglomerativeClustering( distance_threshold = k/30 , linkage = 'single' , n_clusters = None )
    model = model.fit( datanp )
    labels = model.labels_
    
# Affichage clustering
plt.scatter ( f0 , f1 , c = labels , s = 8 )
plt.title ( " Resultat du clustering " )
plt.show()
print (" nb clusters = " , k , " , nb feuilles = " , leaves ,
        " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )


# set the number of clusters

k = 4
tps1 = time.time ()
for k in range(2,20):
    model = cluster.AgglomerativeClustering( linkage = 'single' , n_clusters = k )
    model = model.fit ( datanp )
    
    silhouette.append(metrics.silhouette_score(datanp,model.labels_))
    davies.append(metrics.davies_bouldin_score(datanp,model.labels_))
    calinski.append(metrics.calinski_harabasz_score(datanp,model.labels_))

if(np.argmax(silhouette) == np.argmin(davies) and np.argmin(davies) == np.argmax(calinski) ):
    k = np.argmax(calinski) +2
    print("On a trouvé un gagnant : " + str(k)+ " clusters")

tps2 = time.time ()
labels = model.labels_
kres = model.n_clusters_
leaves = model.n_leaves_

# Affichage clustering
plt.scatter ( f0 , f1 , c = labels , s = 8 )
plt.title ( " Resultat du clustering " )
plt.show()
print (" nb clusters = " , k , " , nb feuilles = " , leaves ,
        " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )