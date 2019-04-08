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

nn = NeuralNetwork.RL([4,2])
past = np.array([0.0, 0.0])

def critic(angle):
    amax = 1.0
    return (amax**2 - angle**2)/amax**2

env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        #action = env.action_space.sample()
        print(observation)
        #action = 1 
        #if observation[1] > 0.0:
        #    action = 0
        pred = NeuralNetwork.RL_Predict(nn, observation)
        action = NeuralNetwork.RL_Action(pred)
        past = NeuralNetwork.RL_AvgAct(past, pred[-1])
        observation, reward, done, info = env.step(action)
        r = critic(observation[1])
        NeuralNetwork.RL_ARP(nn, pred, past, r, 0.01)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

