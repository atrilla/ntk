# -----------------------------------------------------------------------
# File    : test_prnn.py
# Created : 04-Apr-2019
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


import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

# model exp decay
def genseq_decay():
    t = np.array(range(50 + int(50*np.random.rand())))
    A = 0.5 + 5.0*np.random.rand()
    alpha = -0.01 - 0.1*np.random.rand()
    x = A*np.e**(alpha*t)
    x = x.reshape(len(x),1)
    return x

# autoregressive predictor, next sample
# Train
x = []
y = []
print("Generating data...")
for n in xrange(500):
    aux = genseq_decay()
    x.append(aux[:-1])
    y.append(aux[1:])

print("Elman training...")
elm_nn = NeuralNetwork.ELM([1,3,1])
NeuralNetwork.PR_Backprop(elm_nn, x, y, 0.1, 20, 0.001, 
    af=NeuralNetwork.Sigmoid, c='sqerr', arch="ELM")

jalpha = 0.1
print("Jordan training...")
jor_nn = NeuralNetwork.JOR([1,3,1])
NeuralNetwork.PR_Backprop(jor_nn, x, y, 0.1, 20, 0.001, 
    af=NeuralNetwork.Sigmoid, c='sqerr', arch="JOR", alpha=jalpha)

# test seq
test = genseq_decay()
tx = test[:-1]
ty = test[1:]

elm_err = []
context = np.zeros(elm_nn[0].shape[0])
for n in xrange(len(tx)):
    o = NeuralNetwork.PR_Predict(elm_nn, tx[n], context, 
        af=NeuralNetwork.Sigmoid)
    context = NeuralNetwork.ELM_Context(o)
    elm_err.append(ty[n][0] - o[-1][0])

jor_err = []
context = np.zeros(jor_nn[-1].shape[0])
for n in xrange(len(tx)):
    o = NeuralNetwork.PR_Predict(jor_nn, tx[n], context, 
        af=NeuralNetwork.Sigmoid)
    context = NeuralNetwork.JOR_Context(o, alpha=jalpha)
    jor_err.append(ty[n][0] - o[-1][0])


plt.figure()
plt.plot(test, 'k', label="Data")
plt.plot(elm_err, 'b', label="Elman")
plt.plot(jor_err, 'r', label="Jordan")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(loc = "upper right")
plt.title("Partially Recurrent net")

plt.show()

