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
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
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

lam = 0.1

sss = StratifiedShuffleSplit(y, 1, test_size=0.3, random_state=0)
for train_index, test_index in sss:
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	nn = NeuralNetwork.Multilayer([4,4,3])
	print("Training...")
	NeuralNetwork.Backprop(nn, X_train, y_train, lam, 20, 0.1)
	#NeuralNetwork.NumGradDesc(nn, X_train, y_train, lam, 20, 0.1)
	print("Testing...")
	tcost = NeuralNetwork.Cost(nn, X_test, y_test, lam)
	print("J = " + str(tcost))

