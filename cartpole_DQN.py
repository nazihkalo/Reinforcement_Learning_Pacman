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

#Plotting the model and getting summary
from keras.utils.vis_utils import model_to_dot
print(online_network.summary())
# from keras.utils import plot_model
# plot_model(online_network, to_file='online_network.png',show_shapes=True,show_layer_names=True)

#Create replay_memory - a storage of experienced transitions - as deque.
# replay memory is a list of [initial state, action, reward, new state]
# the training will select a random batch from the replay memory
from collections import deque
replay_memory_maxlen = 1_000_000
replay_memory = deque([], maxlen=replay_memory_maxlen)


#Define the epsilon-greedy strategy 
#Choose best Q-value with probability 1-eps and random with proability eps
def epsilon_greedy(q_values, epsilon, n_outputs):
    if random.random() < epsilon:
        return random.randrange(n_outputs)  # random action
    else:
        return np.argmax(q_values)  # q-optimal action

#Set global variables necessary for the learning process.

n_steps = 100_000 # number of times 
warmup = 1_000 # first iterations after random initiation before training starts
training_interval = 4 # number of steps after which dqn is retrained
## the retraining does not necessarily happen on the last 4 steps
# #its just that after 4 steps the online_model is retrained from a random sample of 64 past steps

copy_steps = 2_000 # number of steps after which weights of 
                   # online network copied into target network
gamma = 0.99 # discount rate
batch_size = 64 # size of batch from replay memory 
eps_max = 1.0 # parameters of decaying sequence of eps
eps_min = 0.05
eps_decay_steps = 50_000

if __name__ == '__main__':
    step = 0
    iteration = 0
    done = True

    while step < n_steps:
        if done:
            obs = env.reset()
        iteration += 1
        q_values = online_network.predict(np.array([obs]))[0]  
        epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
        action = epsilon_greedy(q_values, epsilon, nb_actions)
        next_obs, reward, done, info = env.step(action)
        replay_memory.append((obs, action, reward, next_obs, done))
        obs = next_obs

        if iteration >= warmup and iteration % training_interval == 0:
            step += 1
            minibatch = random.sample(replay_memory, batch_size)
            replay_state = np.array([x[0] for x in minibatch])
            replay_action = np.array([x[1] for x in minibatch])
            replay_rewards = np.array([x[2] for x in minibatch])
            replay_next_state = np.array([x[3] for x in minibatch])
            replay_done = np.array([x[4] for x in minibatch], dtype=int)
            target_for_action = replay_rewards + (1-replay_done) * gamma * \
                                        np.amax(target_network.predict(replay_next_state), axis=1)
            target = online_network.predict(replay_state)  # targets coincide with predictions ...
            target[np.arange(batch_size), replay_action] = target_for_action  #...except for targets with actions from replay
            online_network.fit(replay_state, target, epochs=step, verbose=1, initial_epoch=step-1)
            if step % copy_steps == 0:
                target_network.set_weights(online_network.get_weights())

    online_network.save('saved_dqn_model.h5')