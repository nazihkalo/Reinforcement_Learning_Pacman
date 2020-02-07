# Building a DQN model for this custom Ms. Pacman environment

import random
import gc
import time
import numpy as np
import os
import gc
import gym
import json

from keras.models import Sequential, clone_model
from keras.layers import Dense, InputLayer, Flatten, Conv2D, InputLayer
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, TensorBoard
import keras.backend as K
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (9, 9)

## READING IN THE CUSTOM PACMAN ENVIRONMENT AND INSTANTIATING IT

with open('minipacman_test/test_params.json', 'r') as file:
    read_params = json.load(file)
game_params = read_params['params']
env = PacmanGame(**game_params)



#CREATING THE VANILA DQN MODEL FUNCTION - to be used for online & target networks
def create_dqn_model(input_shape, nb_actions, dense_layers, dense_units):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for i in range(dense_layers):
        model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model

# Compile the online network using Adam optimizer and loss function of type `mse`.
input_shape = (len(list(obs['player']) + list(sum(obs['monsters'], ())) + list(sum(obs['diamonds'], ())) + list(sum(obs['walls'], ()))),)
nb_actions = 9
dense_layers = 5
dense_units = 256
online_network = create_dqn_model(input_shape, nb_actions, dense_layers, dense_units)
online_network.summary()
# Clone the online network as target network fixing the same weights as in online network.
target_network = clone_model(online_network)
target_network.set_weights(online_network.get_weights())

#Define the epsilon greedy strategy
def epsilon_greedy(q_values, epsilon, n_outputs):
    if random.random() < epsilon:
        return random.randrange(1,n_outputs)  # random action
    else:
        return np.argmax(q_values)+1          # q-optimal action

#Create deque of replay memory with maxlength of 1M
replay_memory_maxlen = 1_000_000
replay_memory = deque([], maxlen=replay_memory_maxlen)

#Custom function to convert observation to input for DQN
def get_observation(obs):
    return np.array([list(obs['player']) + list(sum(obs['monsters'], ())) + list(sum(obs['diamonds'], ())) + list(sum(obs['walls'], ()))])

#Defining our custom strategy for the test function
def custom_strategy(obs):
    state = get_observation(obs)
    q_values = online_network.predict(state)[0]
    return np.argmax(q_values)+1
    

#Set global variables necessary for the learning process.
name = 'MsPacman_DQN'  # used in naming files (weights, logs, etc)
n_steps = 2000        # total number of training steps (= n_epochs)
warmup = 1000          # start training after warmup iterations
training_interval = 4  # period (in actions) between training steps
save_steps = int(n_steps/10)  # period (in training steps) between storing weights to file
copy_steps = 100       # period (in training steps) between updating target_network weights
gamma = 0.9            # discount rate
skip_start = 90        # skip the start of every game (it's just freezing time before game starts)
batch_size = 64        # size of minibatch that is taken randomly from replay memory every training step
double_dqn = False     # whether to use Double-DQN approach or simple DQN (see above)

# eps-greedy parameters: we slowly decrease epsilon from eps_max to eps_min in eps_decay_steps
eps_max = 1.0
eps_min = 0.05
eps_decay_steps = int(n_steps/2)
learning_rate = 0.001

def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

online_network.compile(optimizer=Adam(learning_rate), loss='mse', metrics=[mean_q])

if not os.path.exists(name):
    os.makedirs(name)
    
weights_folder = os.path.join(name, 'weights')
if not os.path.exists(weights_folder):
    os.makedirs(weights_folder)

csv_logger = CSVLogger(os.path.join(name, 'log.csv'), append=True, separator=';')
tensorboard = TensorBoard(log_dir=os.path.join(name, 'tensorboard'), write_graph=False, write_images=False)

if __name__ == '__main__':
    # counters:
    step = 0          # training step counter (= epoch counter)
    iteration = 0     # frames counter
    episodes = 0      # game episodes counter
    done = True       # indicator that env needs to be reset
    nb_actions = 10

    episode_scores = []  # collect total scores in this list and log it later

    env = PacmanGame(**game_params)

    obs = env.reset()
    while step < n_steps:
        if obs['end_game']:  # game over, restart it
            obs = env.reset()
            score = 0  # reset score for current episode
        
        state = get_observation(obs)

        # Online network evaluates what to do
        iteration += 1
        
        q_values = online_network.predict(state)[0]  # calculate q-values using online network
        # select epsilon (which linearly decreases over training steps):
        epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
        #nb_actions = obs['possible_actions']
        action = epsilon_greedy(q_values, epsilon, nb_actions)
        # Play:
        obs = env.make_action(action) # make action and get results
        reward = obs['reward']
        score += reward
        
        if obs['end_game']:
            episode_scores.append(obs['total_score'])
        
        done = obs['end_game']
        next_state = get_observation(obs)
        # Let's memorize what just happened
        replay_memory.append((state, action, reward, next_state, done))

        if iteration >= warmup and iteration % training_interval == 0:
            # learning branch
            step += 1
            minibatch = random.sample(replay_memory, batch_size)
            replay_state = np.array([x[0][0] for x in minibatch])
            replay_action = np.array([x[1] for x in minibatch])
            replay_rewards = np.array([x[2] for x in minibatch])
            replay_next_state = np.array([x[3][0] for x in minibatch])
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
            target[np.arange(batch_size), replay_action-1] = target_for_action  #...except for targets with actions from replay
            
            # Train online network
            online_network.fit(replay_state, target, epochs=step, verbose=1, initial_epoch=step-1,
                            callbacks=[csv_logger, tensorboard])

            # Periodically copy online network weights to target network
            if step % copy_steps == 0:
                target_network.set_weights(online_network.get_weights())
            # And save weights
            if step % save_steps == 0:
                online_network.save_weights(os.path.join(weights_folder, 'weights_{}.h5f'.format(step)))
                gc.collect()  # also clean the garbage


### BASELINE STRATEGIES FOR COMPARISON
from mini_pacman import test, random_strategy, naive_strategy
random_med = test(strategy=random_strategy, log_file='test_pacman_log_random.json')
naive_med = test(strategy=naive_strategy, log_file='test_pacman_log_naive.json')
custom_med = test(strategy=custom_strategy, log_file='test_pacman_log_custom.json')


print(f'Random Median = {random_med} Naive Median = {naive_med} Custom Median = {custom_med}')