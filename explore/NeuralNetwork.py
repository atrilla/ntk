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
# return list of all neuron output values maintaining layer order
def Predict(nn, x):
	neuron = [x]
	ain = x.tolist()
	ain.insert(0, 1)
	a = np.array(ain)
	for l in nn:
		z = l.dot(a)
		g = Sigmoid(z)
		neuron.append(g)
		ahid = g.tolist()
		ahid.insert(0,1)
		a = np.array(ahid)
	return neuron

# activation function
def Sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

# errors
# nn neural net instance
# o network output prediction
# t targets
def Error(nn, o, t):
	err = []
	err.append(t - o)
	for l in reversed(nn[1:]):
		aux = err[-1].dot(l)
		err.append(aux[1:])
	err.reverse()
	return err

# Learning, training online
# nn neuralnet instance
# x is features
# t is targets
# lam is Tikhonov regularisation
# nepoch is num of iteration over the dataset
# eta is lerning rate
def Backprop(nn, x, t, lam, nepoch, eta):
	for epoch in xrange(nepoch):
		# example fitting
		for xi,ti in zip(x,t):
			o = Predict(nn, xi)
			err = Error(nn, o[-1], ti)
			for lind in xrange(len(nn)):
				l = nn[lind]
				delta = np.ones(l.shape)
				xx = o[lind].tolist()
				xx.insert(0, 1)
				xx = np.array(xx)
				for i in xrange(delta.shape[0]):
					delta[i] *= eta * xx * err[lind][i] * o[lind+1][i] * (1 - o[lind+1][i])
				nn[lind] += delta
		# regularisation
		for l in nn:
			l -= eta * lam * l
		print("J = " + str(Cost(nn, x, t, lam)))

# cost, sqerr
def Cost(nn, x, t, lam):
	sqerr = 0
	for xi,ti in zip(x,t):
		o = Predict(nn, xi)
		err = Error(nn, o[-1], ti)
		sqerr += np.sum(err[-1]**2)
	sqerr = sqerr / t.shape[0]
	reg = 0
	for l in nn:
		aux = l.flatten()
		reg += aux.dot(aux)
	return sqerr
