from keras.models import load_model
import random
import gc
import time
import numpy as np
import json
from mini_pacman import test, random_strategy, naive_strategy, PacmanGame
with open('minipacman_test/test_params.json', 'r') as file:
    read_params = json.load(file)
game_params = read_params['params']
env = PacmanGame(**game_params)


from mini_pacman_DQN import get_observation, custom_strategy, create_dqn_model
# Compile the online network using Adam optimizer and loss function of type `mse`.
input_shape = (32,)
nb_actions = 9
dense_layers = 5
dense_units = 256
dqn_model = create_dqn_model(input_shape, nb_actions, dense_layers, dense_units)
dqn_model.summary()

#Loading model
dqn_model.load_weights('minipacman_test/weights_1600000.h5f')

#Custom function to convert observation to input for DQN
def get_observation(obs):
    return np.array([list(obs['player']) + list(sum(obs['monsters'], ())) + list(sum(obs['diamonds'], ())) + list(sum(obs['walls'], ()))])

#Defining our custom strategy for the test function
def custom_strategy(obs):
    state = get_observation(obs)
    q_values = dqn_model.predict(state)[0]
    return np.argmax(q_values)+1

# ### BASELINE STRATEGIES FOR COMPARISON
# from mini_pacman import test, random_strategy, naive_strategy
# random_med = test(strategy=random_strategy, log_file='test_pacman_log_random.json')
# naive_med = test(strategy=naive_strategy, log_file='test_pacman_log_naive.json')
# custom_med = test(strategy=custom_strategy, log_file='test_pacman_log_custom.json')


# print(f'Random Median = {random_med} Naive Median = {naive_med} Custom Median = {custom_med}')


####### RENDERING TO SEE PERFORMANCE####
episode_history = []
total_history= []
total_scores = []

for game in range(10):
    print(f"Game {game}, let's go!")
    obs = env.reset()
    episode_history.append(obs)
    while not obs['end_game']:
        action = custom_strategy(obs)
        obs = env.make_action(action)
        episode_history.append(obs)
        env.render()
        time.sleep(0.1)

    total_history.append(episode_history)
    total_scores.append(obs['total_score'])
mean_score = np.mean(total_scores)
median_score = np.median(total_scores)

print("Your average score is {}, median is {}. "
        "Do not forget to upload it for submission!".format(mean_score, median_score))