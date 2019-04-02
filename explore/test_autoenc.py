# -----------------------------------------------------------------------
# File    : test_autoenc.py
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

import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

def gensig():
    t = np.array(range(100))
    x = 0.8*np.sin(2.0*np.pi*np.random.rand()*100.0/1000.0*t +
        np.random.rand()*2.0*np.pi)
    return x

# Train
x = []
y = []
for n in xrange(500):
    y.append(gensig())
    x.append(y[-1] + 0.1*np.random.randn(100))

x = np.array(x)
y = np.array(y)

nn = NeuralNetwork.MLP([100,20,100])
NeuralNetwork.MLP_Backprop(nn, x, y, 0.1, 20, 0.01, 
    af=NeuralNetwork.HyperTan)

test = gensig() + 0.1*np.random.randn(100)
pt = NeuralNetwork.MLP_Predict(nn, test, af=NeuralNetwork.HyperTan)

plt.figure()
plt.plot(test, 'b', label="Noisy")
plt.plot(pt[-1], 'r', label="Denoised")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend(loc = "upper right")
plt.title("Denoising autoencoder")

plt.show()

