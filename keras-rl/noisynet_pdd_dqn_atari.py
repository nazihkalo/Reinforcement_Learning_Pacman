from __future__ import division

import argparse
from datetime import datetime

import gym
import keras.backend as K
import numpy as np
from keras.layers import Dense, Flatten, Input, InputLayer
from keras.models import Model, Sequential
from keras.optimizers import Adam
from PIL import Image

from rl_local.layers import NoisyNetDense
from rl_local.agents.dqn import DQNAgent
from rl_local.callbacks import (FileLogger, ModelIntervalCheckpoint,
                                TrainEpisodeLogger)
from rl_local.core import Processor
from rl_local.memory import PrioritizedMemory
from rl_local.policy import GreedyQPolicy

ENV_NAME = 'MsPacman-ram-v0' 

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n 
print("NUMBER OF ACTIONS: " + str(nb_actions))

INPUT_SHAPE = env.reset().shape
WINDOW_LENGTH = 1
nb_steps = 10000



#Standard DQN model architecture, but swapping the Dense classifier layers for the rl.layers.NoisyNetDense version.
def create_model(input_shape = INPUT_SHAPE, nb_actions = nb_actions, dense_layers = 4, dense_units = 256):
    model = Sequential()
    #model.add(InputLayer(input_shape= input_shape))
    model.add(Flatten(input_shape = (1,) + env.observation_space.shape))
    for i in range(dense_layers):
        model.add(Dense(units=dense_units, activation='relu'))
    model.add(NoisyNetDense(units=dense_units, activation='relu'))
    model.add(NoisyNetDense(nb_actions, activation='linear'))
    print(model.summary())
    return model

model = create_model()
#You can use any Memory you want.
memory = PrioritizedMemory(limit=1000000, alpha=.6, start_beta=.4, end_beta=1., steps_annealed=10000000, window_length=WINDOW_LENGTH)

#This is the important difference. Rather than using an E Greedy approach, where
#we keep the network consistent but randomize the way we interpret its predictions,
#in NoisyNet we are adding noise to the network and simply choosing the best value.
policy = GreedyQPolicy()

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, 
                enable_double_dqn=True, enable_dueling_network=True, 
                nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                train_interval=4, delta_clip=1., 
                custom_model_objects={"NoisyNetDense":NoisyNetDense})

#Prioritized Memories typically use lower learning rates
lr = .00025
if isinstance(memory, PrioritizedMemory):
    lr /= 4
dqn.compile(Adam(lr=lr), metrics=['mae', 'mse', 'accuracy'])

time_now = datetime.now().strftime('%Y.%m.%d.%H:%M:%S')
checkpoint_weights_filename = 'noisynet_pdd_dqn_' + ENV_NAME + '_weights_V2_time{}.h5f'.format(time_now)
log_filename = 'noisynet_pdd_dqn_{}_V2_log.json'.format(ENV_NAME)
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
callbacks += [FileLogger(log_filename, interval=100)]
callbacks += [TrainEpisodeLogger()]


dqn.fit(env, nb_steps=nb_steps, visualize=False, verbose=2, callbacks=callbacks)

# After training is done, we save the final weights.
weights_filename = 'noisynet_pdd_dqn_{}_weights_V2_{}.h5f'.format(ENV_NAME, time_now)
dqn.save_weights(weights_filename, overwrite=True)


# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=100, visualize=False)