import numpy as np
import gym

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, InputLayer
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import TrainEpisodeLogger, ModelIntervalCheckpoint, FileLogger

from datetime import datetime


ENV_NAME = 'MsPacman-ram-v0' 


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n -1 #THE keras.rl model adds one for the no action 
input_shape = env.reset().shape
nb_steps = 10000000
# Next, we build a very simple model regardless of the dueling architecture
# if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
# Also, you can build a dueling network by yourself and turn off the dueling network in DQN.

def create_model(input_shape = input_shape, nb_actions = env.action_space.n, dense_layers = 5, dense_units = 256):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    for i in range(dense_layers):
        model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    print(model.summary())
    return model

model = create_model()
# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=nb_steps)
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50000,
               enable_dueling_network=True, dueling_type='avg', 
               enable_double_dqn__ = True,
               target_model_update=10000, 
               policy=policy, train_interval=4, delta_clip=1., 
               batch_size = 256, gamma = 0.999)
dqn.compile(Adam(lr=1e-3), metrics=['mae','mse', 'accuracy'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
time_now = datetime.now().strftime('%Y.%m.%d.%H:%M:%S')
checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_time{}.h5f'.format(time_now)
log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
callbacks += [FileLogger(log_filename, interval=100)]
callbacks += [TrainEpisodeLogger()]


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

tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(ENV_NAME, datetime.now().strftime('%Y.%m.%d.%H.%M.%S')))





dqn.fit(env, nb_steps=nb_steps, visualize=False, verbose=2, callbacks=callbacks)

# After training is done, we save the final weights.
weights_filename = 'duel_dqn_{}_weights_{}.h5f'.format(ENV_NAME, time_now)
dqn.save_weights('duel_dqn_{}_weights_{}.h5f'.format(ENV_NAME, time_now), overwrite=True)



# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)