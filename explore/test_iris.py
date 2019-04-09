# -----------------------------------------------------------------------
# File    : test_iris.py
# Created : 30-Mar-2017
# By      : Alexandre Trilla <alex@atrilla.net>
#
# NTK - Neural Network Toolkit
#
# Copyright (C) 2017 Alexandre Trilla
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

import NeuralNetwork
from sklearn import datasets as dset
from sklearn.utils import shuffle
import numpy as np

iris = dset.load_iris()
t = []
for i in iris.target:
	if i == 0:
		t.append([1,0,0])
	elif i == 1:
		t.append([0,1,0])
	else:
		t.append([0,0,1])

y = np.array(t)
X = iris.data

sx,sy = shuffle(X,y)

lam = 0.1

nn = NeuralNetwork.MLP([4,2,3])
print("Training...")
NeuralNetwork.MLP_Backprop(nn, sx[:100], y[:100], lam, 100, 0.0001)
#NeuralNetwork.MLP_NumGradDesc(nn, X_train, y_train, lam, 20, 0.1)
print("Testing...")
tcost = NeuralNetwork.MLP_Cost(nn, sx[100:], y[100:], lam)
print("J = " + str(tcost))

