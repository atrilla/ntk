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

# observation -> cart pos, cart vel, angle, pole vel at tio
# action -> 0 (push to the left)
#
# https://github.com/openai/gym/wiki/CartPole-v0

import NeuralNetwork
import numpy as np
import gym
import matplotlib.pyplot as plt

# acceleration...
def critic(info):
    phi = info[0]
    dphi = info[0] - info[1]
    aux = info[1] - info[2]
    ddphi = aux - dphi
    return np.max([0.0, 1.0 - 1.0*(phi**2 + 10*dphi**2 + 10*ddphi**2)])

def pid(info):
    phi = info[0]
    dphi = info[0] - info[1]
    iphi = np.mean(info)
#    return 10.0*np.array([phi, dphi, iphi])
    return 10.0*np.array([phi, dphi, iphi])

nn = NeuralNetwork.RL(3)

timeperf = []
env = gym.make('CartPole-v0')
for i_episode in range(100):
    observation = env.reset()
    obsang = [observation[2], 0.0, 0.0]
    end = False
    toptime = 200
    past = np.array([0.0])
    for t in range(toptime):
        env.render()
        #action = env.action_space.sample()
        pred = NeuralNetwork.RL_Predict(nn, pid(obsang))
        action = int(NeuralNetwork.RL_Action(pred))
        past = NeuralNetwork.RL_AvgAct(past, pred[-1])
        observation, reward, done, info = env.step(action)
        obsang.pop()
        obsang.insert(0, observation[2])
        r = critic(obsang)
        print(r)
        NeuralNetwork.RL_ARP(nn, pred, past, r, 5.0)
        if r < 0.6:
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
