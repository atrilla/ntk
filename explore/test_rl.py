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

# maintain pole vertically
def critic(info):
    lmax = 1.0
    vmax = 0.5
    amax = 0.1
    phi = info[0]
    dphi = info[1] - info[0]
    aux = info[3] - info[2]
    ddphi = aux - dphi
    loc = np.max([(lmax**2 - phi**2)/lmax**2, 0.0])
    vel = np.max([(vmax**2 - dphi**2)/vmax**2, 0.0])
    acc = np.max([(amax**2 - ddphi**2)/amax**2, 0.0])
    return (0.5*loc + 0.1*vel + 0.4*acc)

nn = NeuralNetwork.RL(4)
past = np.array([0.0])

timeperf = []
env = gym.make('CartPole-v0')
for i_episode in range(80):
    observation = env.reset()
    obsang = [observation[2], 0.0, 0.0, 0.0]
    end = False
    toptime = 100
    for t in range(toptime):
        env.render()
        #action = env.action_space.sample()
        print(observation)
        pred = NeuralNetwork.RL_Predict(nn, observation)
        action = int(NeuralNetwork.RL_Action(pred))
        past = NeuralNetwork.RL_AvgAct(past, pred[-1])
        observation, reward, done, info = env.step(action)
        obsang.pop()
        obsang.insert(0, observation[2])
        r = critic(obsang)
        NeuralNetwork.RL_ARP(nn, pred, past, r, 5.0)
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
