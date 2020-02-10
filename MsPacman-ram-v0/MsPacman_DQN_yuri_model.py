import matplotlib.pyplot as plt

import numpy as np
import random
import time
import os
import gc
from collections import deque

from keras.models import Sequential, clone_model
from keras.layers import Dense, Flatten, Conv2D, InputLayer
from keras.callbacks import CSVLogger, TensorBoard
from keras.optimizers import Adam
import keras.backend as K

import gym

plt.rcParams['figure.figsize'] = (9, 9)

env = gym.make("MsPacman-ram-v0")
env.action_space.n # actions are integers from 0 to 8

def create_dqn_model(input_shape, nb_actions, dense_layers, dense_units):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for i in range(dense_layers):
        model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model

#Training window - number of frames to train & predict on 
obs_window_maxlen = 4
obs_window = deque([], maxlen=obs_window_maxlen)

input_shape = env.reset().shape
nb_actions = env.action_space.n  # 9
dense_layers = 5
dense_units = 256

online_network = create_dqn_model(input_shape, nb_actions, dense_layers, dense_units)
online_network.summary()

from keras.utils import plot_model
plot_model(online_network, to_file='online_DenseNetwork.png',show_shapes=True,show_layer_names=True)

def epsilon_greedy(q_values, epsilon, n_outputs):
    if random.random() < epsilon:
        return random.randrange(n_outputs)  # random action
    else:
        return np.argmax(q_values)          # q-optimal action


replay_memory_maxlen = 1000000
replay_memory = deque([], maxlen=replay_memory_maxlen)

target_network = clone_model(online_network)
target_network.set_weights(online_network.get_weights())

name = 'MsPacman_DQN'  # used in naming files (weights, logs, etc)
n_steps = 1_000_000        # total number of training steps (= n_epochs)
warmup = 10_000          # start training after warmup iterations
training_interval = 4  # period (in actions) between training steps
save_steps = int(n_steps/20)  # period (in training steps) between storing weights to file
copy_steps = 1000       # period (in training steps) between updating target_network weights
gamma = 0.9            # discount rate
skip_start = 90        # skip the start of every game (it's just freezing time before game starts)
batch_size = 64        # size of minibatch that is taken randomly from replay memory every training step
double_dqn = True     # whether to use Double-DQN approach or simple DQN (see above)
# eps-greedy parameters: we slowly decrease epsilon from eps_max to eps_min in eps_decay_steps
eps_max = 1.0
eps_min = 0.05
eps_decay_steps = int(n_steps/2)

learning_rate = 0.001


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


online_network.compile(optimizer=Adam(learning_rate), loss='mse', metrics=[mean_q])

cwd = os.getcwd()
new_dir = os.path.join(cwd, name)

if not os.path.exists(new_dir):
    os.makedirs(new_dir)
    
weights_folder = os.path.join(new_dir, 'weights')
if not os.path.exists(weights_folder):
    os.makedirs(weights_folder)

csv_logger = CSVLogger(os.path.join(new_dir, 'log.csv'), append=True, separator=';')
tensorboard = TensorBoard(log_dir=os.path.join(new_dir, 'tensorboard'), write_graph=False, write_images=False)

# counters:
step = 0          # training step counter (= epoch counter)
iteration = 0     # frames counter
episodes = 0      # game episodes counter
done = True       # indicator that env needs to be reset

episode_scores = []  # collect total scores in this list and log it later


while step < n_steps:
    if done:  # game over, restart it
        obs = env.reset()
        score = 0  # reset score for current episode
        for skip in range(skip_start):  # skip the start of each game (it's just freezing time before game starts)
            obs, reward, done, info = env.step(0)
            #obs_window.append(obs)
            score += reward
        state = obs #np.array(obs_window)
        episodes += 1

    # Online network evaluates what to do
    iteration += 1
    q_values = online_network.predict(np.array([state]))[0]  # calculate q-values using online network
    # select epsilon (which linearly decreases over training steps):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    action = epsilon_greedy(q_values, epsilon, nb_actions)
    # Play:
    obs, reward, done, info = env.step(action)
    score += reward
    if done:
        episode_scores.append(score)
    next_state = obs
    # Let's memorize what just happened
    replay_memory.append((state, action, reward, next_state, done))
    state = next_state ##obs_window.append(next_state)

    if iteration >= warmup and iteration % training_interval == 0:
        # learning branch
        step += 1
        minibatch = random.sample(replay_memory, batch_size)
        replay_state = np.array([x[0] for x in minibatch])
        replay_action = np.array([x[1] for x in minibatch])
        replay_rewards = np.array([x[2] for x in minibatch])
        replay_next_state = np.array([x[3] for x in minibatch])
        replay_done = np.array([x[4] for x in minibatch], dtype=int)

        # calculate targets (see above for details)
        if double_dqn == False:
            # DQN
            target_for_action = replay_rewards + (1-replay_done) * gamma * \
                                    np.amax(target_network.predict(replay_next_state), axis=1)
        else:
            # Double DQN
            best_actions = np.argmax(online_network.predict(replay_next_state), axis=1)
            target_for_action = replay_rewards + (1-replay_done) * gamma * \
                                    target_network.predict(replay_next_state)[np.arange(batch_size), best_actions]

        target = online_network.predict(replay_state)  # targets coincide with predictions ...
        target[np.arange(batch_size), replay_action] = target_for_action  #...except for targets with actions from replay
        
        # Train online network
        online_network.fit(replay_state, target, epochs=step, verbose=2, initial_epoch=step-1,
                           callbacks=[csv_logger]) #TENSORBOARD TAKEN OUT

        # Periodically copy online network weights to target network
        if step % copy_steps == 0:
            target_network.set_weights(online_network.get_weights())
        # And save weights
        if step % save_steps == 0:
            online_network.save_weights(os.path.join(weights_folder, 'weights_{}.h5f'.format(step)))
            gc.collect()  # also clean the garbage