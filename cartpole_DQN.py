# This section shows a solution of the cartpole problem using a neural network with architecture that follows general 
# [DQN approach developed at Google DeepMind](https://www.nature.com/articles/nature14236).
# The same approach, but with a more complex architecture will also be used to obtain a Q-Learning strategy 
# for the Pac-Man project in the remaining project documents.

import random
import gc
import time
import numpy as np

import gym

from keras.models import Sequential, clone_model
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, TensorBoard
import keras.backend as K

env = gym.make("CartPole-v0")

def create_dqn_model(input_shape, nb_actions): #Input shape is the four numbers [pos,vel,ang,mom] and nb_actions = [0(left),1(right)]
    model = Sequential()
    model.add(Dense(units=8, input_shape=input_shape, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model

# Compile the online network using Adam optimizer and loss function of type `mse`.
# Clone the online network as target network fixing the same weights as in online network.


input_shape = env.observation_space.shape
nb_actions = env.action_space.n
print('input_shape: ',input_shape)
print('nb_actions: ',nb_actions)

online_network = create_dqn_model(input_shape, nb_actions)
online_network.compile(optimizer=Adam(), loss='mse') #once compiled the weigts are initiated
target_network = clone_model(online_network) #now we clone the initiated model with the same weights
target_network.set_weights(online_network.get_weights())

from keras.utils.vis_utils import model_to_dot
print(online_network.summary())

SVG(model_to_dot(online_network).create(prog='dot', format='svg'))