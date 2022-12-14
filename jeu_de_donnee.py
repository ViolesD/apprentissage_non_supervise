#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:07:25 2022

@author: violes
=> fact
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import arff
import pandas as pd

# Parser un fichier de données au format arff
# data est un tableau d'exemples avec pour chacun
# la liste des valeurs des features 
# 
# Dans les jeux de donnée consideres:
# il y a 2 features (dimension 2)
# Ex: [[-0.499261 , -0.0612356],
#      [-1.51369, 0.265446 ],
#       [...]...
#      ]
# Note: chaque exemple du jeu de donnéees contient aussi un numero de cluster
# On retire cette information


path = './artificial/'
# databrut = arff.loadarff(open(path+"xclara.arff",'r'))
# databrut = arff.loadarff(open(path+"atom.arff",'r'))
databrut = arff.loadarff(open(path+"donut1.arff",'r'))
# databrut = arff.loadarff(open(path+"dense-disk-3000.arff",'r'))
# databrut = arff.loadarff(open(path+"insect.arff",'r'))
databrut = arff.loadarff(open(path+"smile1.arff",'r'))
# databrut = arff.loadarff(open(path+"xor.arff",'r'))
# databrut = arff.loadarff(open(path+"twodiamonds.arff",'r'))
# databrut = arff.loadarff(open(path+"twenty.arff",'r'))
# databrut = arff.loadarff(open(path+"square5.arff",'r'))
databrut = arff.loadarff(open(path+"spiral.arff",'r'))
# databrut = arff.loadarff(open(path+"xclara.arff",'r'))
# databrut = arff.loadarff(open(path+"xclara.arff",'r'))

data = [[x[0],x[1]] for x in databrut[0]]

#Affichage een 2D
#Extraire chaque valeur de features pour en faire une liste
#Ex pour f0 = [....]
#Ex pour f1 = [....]
path2 = './dataset-rapport/'
databrut = pd.read_csv(path2+"zz2.txt",sep=" ", encoding="ISO-8859-1", skipinitialspace=True)

data = databrut.to_numpy()



f0= [f[0] for f in data]
f1= [f[1] for f in data]


plt.scatter(f0,f1,s=8)
plt.title('Donnees initiales')
plt.show()