# -----------------------------------------------------------------------
# File    : test_som.py
# Created : 02-Apr-2019
# By      : Alexandre Trilla <alex@atrilla.net>
#
# NTK - Neural Network Toolkit
#
# Copyright (C) 2019 Alexandre Trilla
# -----------------------------------------------------------------------
#
# This file is part of NTK.
#
# NTK is free software: you can redistribute it and/or modify it under
# the terms of the MIT/X11 License as published by the Massachusetts
# Institute of Technology. See the MIT/X11 License for more details.
#
# You should have received a copy of the MIT/X11 License along with
# this source code distribution of NTK (see the COPYING file in the
# root directory).
# If not, see <http://www.opensource.org/licenses/mit-license>.
#
# -----------------------------------------------------------------------


from sklearn import datasets
from sklearn.utils import shuffle
import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
data = shuffle(iris.data)

N=8
som = NeuralNetwork.SOM(4, N)
NeuralNetwork.SOM_Train(som, data, 30, 0.1)

c0 = iris.data[iris.target == 0]
c1 = iris.data[iris.target == 1]
c2 = iris.data[iris.target == 2]

m0 = []
m1 = []
m2 = []

for i in c0:
    bmu, aux = NeuralNetwork.SOM_BMU(som, i)
    coord = NeuralNetwork.SOM_Coord(N, bmu)
    m0.append(coord)

for i in c1:
    bmu, aux = NeuralNetwork.SOM_BMU(som, i)
    coord = NeuralNetwork.SOM_Coord(N, bmu)
    m1.append(coord)

for i in c2:
    bmu, aux = NeuralNetwork.SOM_BMU(som, i)
    coord = NeuralNetwork.SOM_Coord(N, bmu)
    m2.append(coord)

m0 = np.array(m0)
m1 = np.array(m1)
m2 = np.array(m2)

plt.figure()
plt.plot(m0[:,0], m0[:,1], 'or', markersize=12, alpha=0.5, label="Class 0")
plt.plot(m1[:,0], m1[:,1], 'og', markersize=12, alpha=0.5, label="Class 1")
plt.plot(m2[:,0], m2[:,1], 'ob', markersize=12, alpha=0.5, label="Class 2")
plt.xlabel("Neurons")
plt.ylabel("Neurons")
plt.legend(loc = "upper right")
plt.title("SOM with Iris dataset")
plt.show()

