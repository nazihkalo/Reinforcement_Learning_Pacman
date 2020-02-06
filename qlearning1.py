import gym
import numpy as np
env = gym.make('MountainCar-v0')


env.reset()

# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)

# DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

# discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

# print(discrete_os_win_size)

# q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))
# print(q_table.shape)
# print(q_table)

done = False

total_rewards = []

for episode in range(3):
    observation = env.reset()
    episode_reward = 0
    for step in range(1000):
        action = env.action_space.sample()
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        if done:
            break
        total_rewards.append(episode_reward)

# while not done:
#     action = 2 
#     new_state, reward, done, _ = env.step(action)
#     print(reward, new_state)
#     env.render()

print(np.mean(total_rewards))
print(np.var(total_rewards))
print(np.max(total_rewards))
print(np.min(total_rewards))

env.close()

