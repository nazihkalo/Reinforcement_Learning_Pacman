import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential, load_model
from keras.layers import Dense, InputLayer, Flatten, BatchNormalization, Activation, Lambda
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.initializers import VarianceScaling
from keras import backend as K
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import gym
import matplotlib.pyplot as plt
from datetime import datetime
from rl_local.layers import NoisyNetDense
# from rl_local.callbacks import (FileLogger, ModelIntervalCheckpoint,
#                                 TrainEpisodeLogger)
#from rl_local.memory import PrioritizedMemory
#from rl_local.policy import GreedyQPolicy
## PER MEMORY CODE FROM HERE
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb 
from rl_local.memory_PER import Memory
DISCOUNT = 1
REPLAY_MEMORY_SIZE = 500000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 256  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 4  # Terminal states (end of episodes)
MODEL_NAME = '4Layer_256_D1.0_Upd4_2timesreducedLR_newseed1_Mem500k_batch256'
MIN_REWARD = 700  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 5000
skip_start = 90        # skip the start of every game (it's just freezing time before game starts)

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MAX_EPSILON = 1
MIN_EPSILON = 0.001

learning_rate= 0.0000625*2
learning_rate_decay=0.1

#  Stats settings
AGGREGATE_STATS_EVERY = 25  # episodes
SHOW_PREVIEW = False
double_dqn = True

env = gym.make("MsPacman-ram-v0")
input_shape = env.reset().shape

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

#Create dictionary for env
#env_dict = create_env_dict(env)

# Create models folder
if not os.path.isdir('models/{}'.format(MODEL_NAME)):
    os.makedirs('models/{}'.format(MODEL_NAME))

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


# Agent class
class DQNAgent:
    def __init__(self):

        # # Main model
        # model_path = '5Layer_256Neuron__4310.00max__696.60avg__140.00min__1581233860.model'
        # # Create models folder
        # if os.path.exists(os.path.join(os.getcwd(),'models',model_path)):
        #     print('Loading old model weights 2')
        #     self.model = self.create_model()
        #     self.model.load_weights(((os.path.join(os.getcwd(),'models',model_path))))
        # else:
        #     print('No model found. Creating new model')
        #     self.model = self.create_model()

        self.model = self.create_model()
        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        #self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        # Instantiate memory
        self.memory = Memory(REPLAY_MEMORY_SIZE)
     
        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, datetime.now().strftime('%Y.%m.%d.%H.%M.%S')))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        
    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def create_model(self, input_shape = input_shape, nb_actions = env.action_space.n, dense_layers = 2, dense_units = 256):
        model = Sequential()
        model.add(InputLayer(input_shape))
        #model.add(Lambda(lambda x: x/255, name = 'normalizer'))
        for i in range(dense_layers):
            model.add(Dense(units=dense_units,activation='relu', kernel_initializer=VarianceScaling(scale=2.0)))
        # model.add(NoisyNetDense(units= dense_units, activation='relu'))
        # model.add(NoisyNetDense(units= dense_units, activation='relu'))
        model.add(NoisyNetDense(units= dense_units, activation='relu'))
        model.add(NoisyNetDense(units = nb_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate), metrics=['accuracy', 'mse'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    # def update_replay_memory(self, transition):
    #     self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # # Start training only if certain number of samples is already saved
        # if len(self.memory) < MIN_REPLAY_MEMORY_SIZE:
        #     return

        # Get a minibatch of random samples from memory replay table
        tree_idx, batch, ISWeights_mb = self.memory.sample(MINIBATCH_SIZE)
        #minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        replay_state = np.array([each[0][0] for each in batch])
        replay_action = np.array([each[0][1] for each in batch])
        replay_rewards = np.array([each[0][2] for each in batch]) 
        replay_next_state = np.array([each[0][3] for each in batch])
        replay_done = np.array([each[0][4] for each in batch])

        
        # Get current states from minibatch, then query NN model for Q values
        # replay_state = np.array([x[0] for x in minibatch])
        # replay_action = np.array([x[1] for x in minibatch])
        # replay_rewards = np.array([x[2] for x in minibatch])
        # replay_next_state = np.array([x[3] for x in minibatch])
        # replay_done = np.array([x[4] for x in minibatch], dtype=int)

        # calculate targets (see above for details)
        if double_dqn == False:
            # DQN
            target_for_action = replay_rewards + (1-replay_done) * DISCOUNT * \
                                    np.amax(self.target_model.predict(replay_next_state/255), axis=1)
        else:
            # Double DQN
            best_actions = np.argmax(self.model.predict(replay_next_state/255), axis=1) 
            #A_p
            target_for_action = replay_rewards + (1-replay_done) * DISCOUNT * \
                                    self.target_model.predict(replay_next_state/255)[np.arange(MINIBATCH_SIZE), best_actions]
            #y[A]

        target = self.model.predict(replay_state/255)  # targets coincide with predictions ...
        
        #GET ABS TD ERROR FOR PER
        #absolute_errors = abs(target[replay_action] - target_for_action)
        #TDerrors = y_old[A] - y[A] = online netword prediction - target network
        absolute_errors = abs(target[np.arange(MINIBATCH_SIZE), replay_action] - target_for_action)

        target[np.arange(MINIBATCH_SIZE), replay_action] = target_for_action  #...except for targets with actions from replay
        # Train online network
        self.model.fit(replay_state/255, target, 
                        batch_size=MINIBATCH_SIZE, 
                        verbose=0, 
                        shuffle=False, 
                        callbacks=[self.tensorboard] if terminal_state else None)
            
        
        # Update priority
        self.memory.batch_update(tree_idx, absolute_errors)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]
    


agent = DQNAgent()


env.reset()
done = False
for i in range(MIN_REPLAY_MEMORY_SIZE):
    
    if i == 0:
        # First we need a state
        state = env.reset()
        for skip in range(skip_start):  # skip the start of each game (it's just freezing time before game starts)
            state, reward, done, info = env.step(0)
        action = random.randrange(env.action_space.n)
        next_state, reward, done, info = env.step(action) 
    else:
        action = random.randrange(env.action_space.n)  # select random action
        state, reward, done, info = env.step(action)     
    # If we're dead
    if done:
        # We finished the episode
        next_state = np.zeros(state.shape)
        
        # Add experience to memory
        experience = state, action, reward, next_state, done
        agent.memory.store(experience)
        
        # Start a new episode
        # First we need a state
        state = env.reset()
        for skip in range(skip_start):  # skip the start of each game (it's just freezing time before game starts)
            state, reward, done, info = env.step(0)
    else:
        # Add experience to memory
        experience = state, action, reward, next_state, done
        agent.memory.store(experience)
        
        # Our state is now the next_state
        state = next_state



# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode
    

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()
    #before_add = create_before_add_func(env)
    
    for skip in range(skip_start):  # skip the start of each game (it's just freezing time before game starts)
        current_state, reward, done, info = env.step(0)
        episode_reward += reward

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        # if np.random.random() > epsilon:
        #     # Get action from Q table
        #     action = np.argmax(agent.get_qs(current_state))
        # else:
        #     # Get random action
        #     action = np.random.randint(0, env.action_space.n)
        action = np.argmax(agent.get_qs(current_state))
        new_state, reward, done, info = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        #agent.update_replay_memory((current_state, action, reward, new_state, done))
        experience = current_state, action, reward, new_state, done
        agent.memory.store(experience)

        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    print('Episode Reward = {}. Step Count = {}'.format(episode_reward, step))
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        med_reward = np.median(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, 
                                        reward_min=min_reward, 
                                        reward_max=max_reward, 
                                        #epsilon=epsilon,
                                        median = med_reward,
                                        episode_reward = episode_reward)
        time_now = datetime.now().strftime('%Y.%m.%d.%H.%M.%S')
        print("MEDIAN = {}. MAX = {}. MIN {}".format(med_reward, max_reward, min_reward))
        # agent.model.save('models/{0}/{1:_>7.2f}med_{2:_>7.2f}avg_{3:_>7.2f}min__{4}.model'\
        #         .format(MODEL_NAME, med_reward, average_reward, min_reward, time_now))
        
        # Save model, but only when average_reward is greater or equal a set value
        if average_reward >= MIN_REWARD:
            agent.model.save('models/{0}/{1:_>7.2f}med_{2:_>7.2f}avg_{3:_>7.2f}min__{4}.model'\
                .format(MODEL_NAME, med_reward, average_reward, min_reward, time_now))

    # # Decay epsilon
    # if epsilon > MIN_EPSILON:
    #     # epsilon *= EPSILON_DECAY
    #     # epsilon = max(MIN_EPSILON, epsilon)
    #     epsilon = max(MIN_EPSILON, MAX_EPSILON - (MAX_EPSILON-MIN_EPSILON) * (episode/(0.75*EPISODES)))
