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

t = np.array(t)

nn = NeuralNetwork.Multilayer([4,5,4,3])
NeuralNetwork.Backprop(nn, iris.data, t, 0, 200, 0.01)

