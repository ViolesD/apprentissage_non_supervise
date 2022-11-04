#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:49:34 2022

@author: violes
"""



import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import cluster
from sklearn import metrics

from scipy.io import arff

import kmedoids

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances



path = './artificial/'
#databrut = arff.loadarff(open(path+"xclara.arff",'r'))
#databrut = arff.loadarff(open(path+"square1.arff",'r'))
#databrut = arff.loadarff(open(path+"sizes1.arff",'r'))
#databrut = arff.loadarff(open(path+"simplex.arff",'r'))

databrut = arff.loadarff(open(path+"smile3.arff",'r'))
databrut = arff.loadarff(open(path+"banana.arff",'r'))
databrut = arff.loadarff(open(path+"complex9.arff",'r'))


datanp = [[x[0],x[1]] for x in databrut[0]]

f0= [f[0] for f in datanp]
f1= [f[1] for f in datanp]

plt.scatter(f0,f1,s=8)
plt.title('Donnees initiales')
plt.show()


tps1 = time.time()
silhouette =[]
davies = []
calinski = []
distmatrix = []

# distmatrix=euclidean_distances( datanp )
distmatrix=manhattan_distances(datanp)

for k in range(2,20):
# k=3
    fp = kmedoids.fasterpam(distmatrix, k)
    
    silhouette.append(metrics.silhouette_score(datanp,fp.labels))
    davies.append(metrics.davies_bouldin_score(datanp,fp.labels))
    calinski.append(metrics.calinski_harabasz_score(datanp,fp.labels))
    
   
if(np.argmax(silhouette) == np.argmin(davies) and np.argmin(davies) == np.argmax(calinski) ):
    k = np.argmax(calinski) +2
    print("On a trouvé un gagnant : " + str(k)+ " clusters")
    

else:
    k=9;
    

fp = kmedoids.fasterpam(distmatrix, int(k))

model = cluster.KMeans ( n_clusters=k , init ='k-means++')
model.fit( datanp )



randscore = metrics.rand_score(fp.labels,model.labels_)
print("Similarité rand_score : ", randscore)

mutual = metrics.mutual_info_score(fp.labels,model.labels_)
print("Similarité mutual_information : ", mutual )        

tps2 = time.time( )
iter_kmed = fp.n_iter
labels_kmed = fp.labels




print( "Loss with FasterPAM : " , fp.loss )
plt.scatter( f0 , f1 , c=labels_kmed , s =8)
plt.title( "Donnees apres clustering KMedoids " )
plt.show ( )
print("nb clusters =" ,k , " , nb iter =" , iter_kmed , " ,...  runtime = " , round ( ( tps2 - tps1 ) * 1000 , 2 ) , " ms" )