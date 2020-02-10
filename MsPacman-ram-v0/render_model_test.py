import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
import gc

from keras.models import Sequential, clone_model
from keras.layers import Dense, Flatten, Conv2D, InputLayer
from keras.callbacks import CSVLogger, TensorBoard
from keras.optimizers import Adam
import keras.backend as K

import gym

plt.rcParams['figure.figsize'] = (9, 9)

env = gym.make("MsPacman-ram-v0")
input_shape = env.reset().shape
input_shape = env.reset().shape

def create_model(input_shape = input_shape, nb_actions = env.action_space.n, dense_layers = 5 , dense_units = 256):
    model = Sequential()
    model.add(InputLayer(input_shape))
    for i in range(dense_layers):
        model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model
online_network = create_model()

from keras.models import load_model
online_network.load_weights('5Layer_256Neuron__1960.00max__383.40avg__120.00min__1581202550.model')