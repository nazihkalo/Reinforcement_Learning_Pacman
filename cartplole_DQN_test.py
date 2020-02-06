from keras.models import load_model
import gym 
import random
import gc
import time
import numpy as np
from cartpole_DQN import *
from keras.models import Sequential, clone_model
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, TensorBoard
import keras.backend as K

env = gym.make("CartPole-v0")
dqn_model = load_model('saved_dqn_model.h5')


def test_dqn(env, n_games, model, nb_actions, eps=0.05, render=False, sleep_time=0.01):
    scores = []
    for i in range(n_games):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            q_values = model.predict(np.array([obs]))[0]
            action = epsilon_greedy(q_values, eps, nb_actions)
            obs, reward, done, info = env.step(action)
            score += reward
            if render:
                env.render()
                time.sleep(sleep_time)
        scores.append(score)
    return scores
nb_actions = env.action_space.n

# set render=True in order to see how good (or bad) is the trained Q-network
scores = test_dqn(env, 10, dqn_model, nb_actions, eps=0.01, render=True)

#Calculate summary statistics for earned scores 
nn_mean,nn_std,nn_min,nn_max, nn_median = np.mean(scores), np.std(scores), \
                               np.min(scores), np.max(scores), np.median(scores)

#Compare the summary statistics with the random benchmark and the manual strategy developed previously.
from tabulate import tabulate
all_summaries = np.array([['Random',rand_mean,rand_std,rand_min,rand_max, rand_median], \
                          ['Basic',basic_mean,basic_std,basic_min,basic_max, basic_median],\
                          ['DQNet',nn_mean,nn_std,nn_min,nn_max, nn_median]])
headers = ['Policy','Mean','Std','Min','Max','Median']
summary_table = tabulate(all_summaries,headers)
print(summary_table)