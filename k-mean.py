#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:27:13 2022

@author: violes
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import cluster
from sklearn import metrics
# repris de l'exemple d'avant
from scipy.io import arff
import pandas as pd

path = './artificial/'
# databrut = arff.loadarff(open(path+"xclara.arff",'r'))
# databrut = arff.loadarff(open(path+"square1.arff",'r'))
# databrut = arff.loadarff(open(path+"sizes1.arff",'r'))
# databrut = arff.loadarff(open(path+"simplex.arff",'r'))

# databrut = arff.loadarff(open(path+"smile3.arff",'r'))
# databrut = arff.loadarff(open(path+"banana.arff",'r'))
# databrut = arff.loadarff(open(path+"complex9.arff",'r'))
#datanp = [[x[0],x[1]] for x in databrut[0]]

#pour les données spéciales
path2 = './dataset-rapport/'
databrut = pd.read_csv(path2+"zz1.txt",sep=" ", encoding="ISO-8859-1", skipinitialspace=True)


datanp = databrut.to_numpy()

f0= [f[0] for f in datanp]
f1= [f[1] for f in datanp]

plt.scatter(f0,f1,s=8)
plt.title('Donnees initiales')
plt.show()

#
# Les donnees sont dans datanp ( 2 dimensions )
# f 0 : valeurs sur la premiere dimension
# f 1 : valeur sur la deuxieme dimension
#

print( " Appel KMeans pour une valeur fixee de k" )

tps1 = time.time( )

silhouette =[]
davies = []
calinski = []

for k in range(2,20):
    model = cluster.KMeans ( n_clusters=k , init ='k-means++')
    model.fit( datanp )

    silhouette.append(metrics.silhouette_score(datanp,model.labels_))
    davies.append(metrics.davies_bouldin_score(datanp,model.labels_))
    calinski.append(metrics.calinski_harabasz_score(datanp,model.labels_))

if(np.argmax(silhouette) == np.argmin(davies) and np.argmin(davies) == np.argmax(calinski) ):
    # on cherche le k précédent  
    k = np.argmax(calinski) +2
    print("On a trouvé un gagnant : k=" + str(k))

elif (np.argmin(davies) == np.argmax(calinski)):
    k = np.argmax(calinski) +2
    print("On a trouvé un semi-gagnant : k=" + str(k))
elif (np.argmin(davies) == np.argmax(silhouette)):
    k = np.argmax(silhouette) +2
    print("On a trouvé un semi-gagnant : k=" + str(k))   
elif (np.argmax(silhouette) == np.argmax(calinski)):
    k = np.argmax(calinski) +2
    print("On a trouvé un semi-gagnant : k=" + str(k))
else:
    print("Aucun candidat sort du lot")
    #choix par défaut en fonction de ce que l'on veut plus voir
    k=2;

#on relance avec le meilleur modèle 
model = cluster.KMeans ( n_clusters=k , init ='k-means++')
model.fit( datanp )
tps2 = time.time( )

labels = model.labels_
iteration = model.n_iter_

plt.scatter( f0 , f1 , c=labels , s =8)
plt.title( "Donnees apres clustering Kmeans, k=2 " )
plt.show()
print( "nb clusters=" ,k , " , nb iter=",iteration, " , ... ... runtime = " , round ( ( tps2 - tps1) * 1000 , 2 ) , " ms " )