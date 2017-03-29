# -----------------------------------------------------------------------
# File    : NeuralNetwork.py
# Created : 29-Mar-2017
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

# Multilayer Perceptron, sigmoid activation

import numpy as np

# inilay, list with layer units, eg, [2,4,1], hidden layer with 4 units
# return neural net instance
def Multilayer(inilay):
	layer = []
	for i,o in zip(inilay, inilay[1:]):
		layer.append(InitWeight(i+1, o))
	return layer

# L is the neuron size of the layers
def InitWeight(Lin, Lout):
	epsilon = np.sqrt(6) / np.sqrt(Lin + Lout);
	return np.random.uniform(-epsilon, epsilon, [Lout, Lin])

# Feed forward
# x is ndarray
# nn is neural net instance
def Predict(nn, x):
	ain = x.tolist()
	ain.insert(0, 1)
	a = np.array(ain)
	for l in nn:
		z = l.dot(a)
		g = Sigmoid(z)
		ahid = g.tolist()
		ahid.insert(0,1)
		a = np.array(ahid)
	return a[1:]

# activation function
def Sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))
