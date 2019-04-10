# -----------------------------------------------------------------------
# File    : test_rl.py
# Created : 03-Apr-2019
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

# python 3 here

# observation -> x, phi, delta x, delta phi
# action -> 0 (push to the left)

import NeuralNetwork
import numpy as np
import gym
import matplotlib.pyplot as plt

nn = NeuralNetwork.RL(4)
past = np.array([0.0])

# maintain pole vertically
def critic(angle, speed, accel):
    amax = 1.0
    omax = 1.0
    accmax = 1.0
    alph = np.max([(amax**2 - angle**2)/amax**2, 0.0])
    omeg = np.max([(omax**2 - speed**2)/omax**2, 0.0])
    acce = np.max([(accmax**2 - accel**2)/accmax**2, 0.0])
    return (alph + omeg + acce)/3.0

timeperf = []
env = gym.make('CartPole-v0')
for i_episode in range(50):
    observation = env.reset()
    end = False
    toptime = 100
    for t in range(toptime):
        env.render()
        #action = env.action_space.sample()
        print(observation)
        #action = 1 
        #if observation[1] > 0.0:
        #    action = 0
        pred = NeuralNetwork.RL_Predict(nn, observation)
        action = int(NeuralNetwork.RL_Action(pred))
        past = NeuralNetwork.RL_AvgAct(past, pred[-1])
        oldobs = observation
        observation, reward, done, info = env.step(action)
        r = critic(observation[1], observation[3], 
                observation[3] - oldobs[3])
        NeuralNetwork.RL_ARP(nn, pred, past, r, 3.0)
        if done:
            end = True
            timeperf.append(t)
            print("Episode finished after {} timesteps".format(t+1))
            break
    if not end:
        timeperf.append(toptime)
env.close()

plt.figure()
plt.plot(timeperf)
plt.xlabel("Epoch")
plt.ylabel("Time upright")
plt.title("Cart-pole balancing performance")
plt.show()
