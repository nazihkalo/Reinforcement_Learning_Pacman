import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('MountainCar-v0')


env.reset()

# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)


DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
SHOW_EVERY = 500


q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))

aggr_ep_rewards = {'ep': [], 'avg' : [], 'med':[], 'min':[], 'max':[]}
ep_rewards = []

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/ discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

discrete_state = get_discrete_state(env.reset())

for episode in range(EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        #print(episode, reward, new_state)
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state+ (action,)]
            #Get new Q value
            new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE * (reward + DISCOUNT + max_future_q)
            #Update discrete state's/action pair's Q Value in the Q table
            q_table[discrete_state+(action,)] = new_q

        elif new_state[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action,)] = 0 
        
        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    
    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY: 
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        median_reward = np.median(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['med'].append(median_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f'Episode: {episode} avg: {average_reward}, med {median_reward}, min {min(ep_rewards[-SHOW_EVERY:])}, max{max(ep_rewards[-SHOW_EVERY:])}')


env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label = 'avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['med'], label = 'med')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label = 'min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label = 'max')
plt.legend(loc = 4)
plt.show()
