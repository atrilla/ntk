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


import numpy as np
import time

# Multilayer Perceptron (MLP), bias is automatically managed
# inilay, list with layer units, eg, [2,4,1], hidden layer with 4 units
# return neural net instance
def MLP(inilay):
    layer = []
    for i,o in zip(inilay, inilay[1:]):
        layer.append(MLP_InitWeight(i+1, o))
    return layer

# L is the neuron size of the layers
# return transition matrix
def MLP_InitWeight(Lin, Lout):
    epsilon = np.sqrt(6) / np.sqrt(Lin + Lout);
    return np.random.uniform(-epsilon, epsilon, [Lout, Lin])

# Logistic activation function
def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Recommended tanh activation function
def HyperTan(x):
    return 1.7159 * np.tanh(2.0/3.0 * x)

# Rectifier, for deep learning
def ReLU(x):
    valtr = []
    if isinstance(x, np.ndarray):
        for i in x:
            aux = 0.0
            if i > 0.0:
                valtr.append(i)
            else:
                valtr.append(0.0)
        valtr = np.array(valtr)
    else:
        if x > 0.0:
            valtr = x
        else:
            valtr = 0.0
    return valtr

# Feed forward
# x is ndarray, F features (input layer size)
# nn is neural net instance
# af is activation function
# return list of all neuron output values (ndarray) maintaining layer order
def MLP_Predict(nn, x, af=Sigmoid):
    neuron = [x]
    ain = x.tolist()
    ain.insert(0, 1)
    a = np.array(ain)
    for l in nn:
        z = l.dot(a)
        g = af(z)
        neuron.append(g)
        ahid = g.tolist()
        ahid.insert(0,1)
        a = np.array(ahid)
    return neuron

# Network errors
# nn neural net instance
# o network output prediction, ndarray with O outputs
# t target, ndarray with O outputs
# c cost function, 'sqerr' (default), 'xent'
# return list of all neuron error values (ndarray) maintaining layer order
def MLP_Error(nn, o, t, c='sqerr'):
    err = [t - o]
    if c=='xent':
        err = [-((1.0-t)/(1.0-o) - t/o)]
    for l in reversed(nn[1:]):
        aux = err[-1].dot(l)
        err.append(aux[1:])
    err.reverse()
    return err

# Learning, training online
# nn neural net instance
# x is examples, ndarray (N,F), N instances, F features
# t is targets, ndarray (N,O), N instances, O outputs
# lam is Tikhonov regularisation, float
# nepoch is num of iteration over the dataset, int
# eta is lerning rate, float
# af is activation function
# c cost function, 'sqerr', 'xent' (af must be sigmoid, default)
# network weights are adjusted
def MLP_Backprop(nn, x, t, lam, nepoch, eta, af=Sigmoid, c='sqerr'):
    A = 1.7159
    B = 2.0 / 3.0
    tics = time.time()
    for epoch in xrange(nepoch):
        # example fitting
        for xi,ti in zip(x,t):
            o = MLP_Predict(nn, xi, af=af)
            err = MLP_Error(nn, o[-1], ti, c=c)
            for lind in xrange(len(nn)):
                l = nn[lind]
                delta = np.ones(l.shape)
                xx = o[lind].tolist()
                xx.insert(0, 1)
                xx = np.array(xx)
                for i in xrange(delta.shape[0]):
                    if af == Sigmoid:
                        delta[i] *= eta * xx * err[lind][i] * o[lind+1][i] * (1 - o[lind+1][i])
                    elif af == HyperTan:
                        delta[i] *= eta * err[lind][i] * ( 1.0/A * (A**2 - (o[lind+1][i])**2) * B * xx)
                    elif af == ReLU:
                        if o[lind+1][i] > 0.0:
                            delta[i] *= eta * xx * err[lind][i]
                        else:
                            delta[i] = 0.0
                l += delta
        # regularisation
        M = x.shape[0]
        for l in nn:
            aux = l[:,0]
            l -= eta * lam / M * l
            l[:,0] = aux
        print("J(" + str(epoch) + ") = " + str(MLP_Cost(nn, x, t, lam, af=af, c=c)))
    print("Elapsed time = " + str(time.time() - tics) + " seconds")

# cost, sqerr
# x is examples, ndarray (N,F), N instances, F features
# t is targets, ndarray (N,O), N instances, O outputs
# lam is Tikhonov regularisation, float
# af is activation function
# c cost function, 'sqerr' (default), 'xent'
def MLP_Cost(nn, x, t, lam, af=Sigmoid, c='sqerr'):
    fit = 0
    if c=='sqerr':
        sqerr = 0
        for xi,ti in zip(x,t):
            o = MLP_Predict(nn, xi, af=af)
            err = MLP_Error(nn, o[-1], ti, c=c)
            sqerr += np.sum(err[-1]**2)
        fit = sqerr / t.shape[0]
    if c=='xent':
        xent = 0
        for xi,ti in zip(x,t):
            o = MLP_Predict(nn, xi, af=af)
            xent += np.sum(-(ti*np.log(o[-1]) + (1.0 - ti)*np.log(1.0 - o[-1])))
        fit = xent
    reg = 0
    M = x.shape[0]
    for l in nn:
        aux = l.flatten()
        reg += (lam/(2.0 * M)) * aux.dot(aux)
    return fit + reg

# Learning, training batch
# nn neuralnet instance
# x is examples, ndarray (N,F), N instances, F features
# t is targets, ndarray (N,O), N instances, O outputs
# lam is Tikhonov regularisation, float
# nepoch is num of iteration over the dataset, int
# eta is lerning rate, float
# af is activation function
# network weights are adjusted
def MLP_NumGradDesc(nn, x, t, lam, nepoch, eta, af=Sigmoid):
    incr = 0.0001
    tics = time.time()
    for epoch in xrange(nepoch):
        for l in nn:
            for w in l:
                ref = w
                w += incr
                plus = MLP_Cost(nn, x, t, lam, af=af)
                w -= 2.0*incr
                minus = MLP_Cost(nn, x, t, lam, af=af)
                w += incr
                w -= eta*(plus - minus)/(2.0*incr)
        # regularisation
        M = x.shape[0]
        for l in nn:
            aux = l[:,0]
            l -= eta * lam / M * l
            l[:,0] = aux
        print("J(" + str(epoch) + ") = " + str(MLP_Cost(nn, x, t, lam, af=af)))
    print("Elapsed time = " + str(time.time() - tics) + " seconds")

# Time-Delay Neural Network
# alpha, float, filter spreading factor
# inilay, list with layer units, eg, [2,4,1], hidden layer with 4 units
# return neural net instance (TDNN)
def TDNN(alpha, inilay):
    nn = MLP(inilay)
    G = []
    N = inilay[0]
    for i in xrange(N):
        G.append(TDNN_Filter(N, alpha, i))
    G = np.array(G)
    return [G, nn]

# Spreading filter
# N, int, sequence size
# alpha, float, filter spreading factor
# d, int, buffer delay
# return filter
def TDNN_Filter(N, alpha, d):
    n = np.array(range(N)) + 1.0
    s = np.sum((n/(d+1.0))**alpha * np.exp(alpha*(1.0-n)/(d+1.0)))
    g = (n/(d+1.0))**alpha * np.exp(alpha*(1.0-n)/(d+1.0))/s
    return g

def TDNN_Predict(tdnn, x, af=Sigmoid):
    g = tdnn[0]
    mlp = tdnn[1]
    gx = g.dot(x)
    return MLP_Predict(mlp, gx, af=af)

def TDNN_Backprop(tdnn, x, t, lam, nepoch, eta, af=Sigmoid):
    g = tdnn[0]
    mlp = tdnn[1]
    gx = x.dot(g.transpose())
    MLP_Backprop(mlp, gx, t, lam, nepoch, eta, af=af)

def TDNN_Cost(tdnn, x, t, lam, af=Sigmoid):
    g = tdnn[0]
    mlp = tdnn[1]
    gx = x.dot(g.transpose())
    return MLP_Cost(tdnn, x, t, lam, af=af)

# Self-Organising Map
# I, int, input network size
# N, int, map size (NxN), network output
# return neural net instance (SOM)
def SOM(I, N):
    nn = MLP_InitWeight(I, N**2)
    topo = SOM_Topology(N)
    return [nn, topo]

# Create neuron index topology map
# N, int, map size (NxN), network output
def SOM_Topology(N):
    ind = np.array(range(N**2))
    ind = ind.reshape((N,N))
    return ind

# Kohonen training
# nn SOM instance
# x is examples, ndarray (N,F), N instances, F features
# epoch, int, number of rounds
# eta is lerning rate, float
# network weights are adjusted
def SOM_Train(nn, x, epoch, eta):
    for round in xrange(epoch):
        print("SOM training round: " + str(round))
        for i in x:
            bmu,aux = SOM_BMU(nn, i)
            nn[0][bmu] += eta*(i - nn[0][bmu])
            neigh = SOM_Neighba(nn[1], bmu)
            for n in neigh:
                nn[0][n] += eta*(i - nn[0][n])*0.5

# return Euclidean distance between two ndarrays
def euclid(x,y):
    d = x - y
    return np.sqrt(np.sum(d**2))

# find best-matching unit's topology ID
# nn is SOM instance
# x is idarray input
# return id, int
# return euclid distance
def SOM_BMU(nn, x):
    best = euclid(nn[0][0], x)
    mapid = 0
    for i in xrange(2,nn[0].shape[0]):
        ed = euclid(nn[0][i], x)
        if ed < best:
            best = ed
            mapid = i
    return mapid, best

# return list of neuron ID's
def SOM_Neighba(topo, bmu):
    n = []
    N = topo.shape[0]
    # left
    c = bmu - 1
    if c >= 0:
        if c%N == bmu%N - 1:
            n.append(c)
    # right
    c = bmu + 1
    if c <= N**2 - 1:
        if c%N == bmu%N + 1:
            n.append(c)
    # up
    c = bmu - N
    if c >= 0:
        n.append(c)
    # down
    c = bmu + N
    if c <= N**2 - 1:
        n.append(c)
    #
    return n

# return list of neuron map corrdinates x,y
def SOM_Coord(N, bmu):
    x = bmu%N
    y = (bmu + N)/N
    return [x,y]

# Elman network
# inilay, list with layer units, eg, [I,H,O]
# return Elman neural net instance
def ELM(inilay):
    return MLP([inilay[0] + inilay[1], inilay[1], inilay[2]])

# nn is a partially recurrent neural net instance
# x is ndarray, I features (input layer size), seq item at a given time
# c is ndarray, context
# af is activation function
# return list of all neuron output values (ndarray) maintaining layer order
def PR_Predict(nn, x, c, af=Sigmoid):
    aux = np.append(c,x)
    return MLP_Predict(nn, aux, af=af)

# p, ndarray, Elman network prediction
# returns context (i.e., the hidden layer)
def ELM_Context(p):
    return p[1]

# Partially Recurrent Learning, training online
# nn PR neural net instance
# arch, string, bet architecture: "ELM", "JOR".
# x is examples, list of ndarray instances (N sequences); (T,F), T seq
#   samples, F features
# t is targets, list of ndarray instances (N sequences); (T,O), T seq
#   samples, O outputs. O can be "None", and the nwt does not learn with
#   that sequence sample. The sequence length T is arbitrary.
# lam is Tikhonov regularisation, float
# nepoch is num of iteration over the dataset, int
# eta is lerning rate, float
# af is activation function
# c cost function, 'sqerr', 'xent' (af must be sigmoid, default)
# network weights are adjusted
def PR_Backprop(nn, x, t, lam, nepoch, eta, af=Sigmoid, c='sqerr',
    arch="ELM"):
    A = 1.7159
    B = 2.0 / 3.0
    tics = time.time()
    for epoch in xrange(nepoch):
        # example fitting
        for xi,ti in zip(x,t):
            for n in zip(xi,ti):
            #######################################################
            # Elman default
            o = PR_Predict(nn, xi, np.zeros(len(nn[1])), af=af)
            if ti is not None:
            err = MLP_Error(nn, o[-1], ti, c=c)
            for lind in xrange(len(nn)):
                l = nn[lind]
                delta = np.ones(l.shape)
                xx = o[lind].tolist()
                xx.insert(0, 1)
                xx = np.array(xx)
                for i in xrange(delta.shape[0]):
                    if af == Sigmoid:
                        delta[i] *= eta * xx * err[lind][i] * o[lind+1][i] * (1 - o[lind+1][i])
                    elif af == HyperTan:
                        delta[i] *= eta * err[lind][i] * ( 1.0/A * (A**2 - (o[lind+1][i])**2) * B * xx)
                    elif af == ReLU:
                        if o[lind+1][i] > 0.0:
                            delta[i] *= eta * xx * err[lind][i]
                        else:
                            delta[i] = 0.0
                l += delta
        # regularisation
        M = x.shape[0]
        for l in nn:
            aux = l[:,0]
            l -= eta * lam / M * l
            l[:,0] = aux
        print("J(" + str(epoch) + ") = " + str(MLP_Cost(nn, x, t, lam, af=af, c=c)))
    print("Elapsed time = " + str(time.time() - tics) + " seconds")


