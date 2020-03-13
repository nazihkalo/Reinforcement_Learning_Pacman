import argparse
import json

import matplotlib.pyplot as plt


def visualize_log(filename, figsize=None, output=None):
    with open(filename, 'r') as f:
        data = json.load(f)
    if 'episode' not in data:
        raise ValueError('Log file "{}" does not contain the "episode" key.'.format(filename))
    episodes = data['episode']

    # Get value keys. The x axis is shared and is the number of episodes.
    keys = sorted(list(set(data.keys()).difference(set(['episode']))))

    if figsize is None:
        figsize = (15., 5. * len(keys))
    f, axarr = plt.subplots(len(keys), sharex=True, figsize=figsize)
    for idx, key in enumerate(keys):
        axarr[idx].plot(episodes, data[key])
        axarr[idx].set_ylabel(key)
    plt.xlabel('episodes')
    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(output)

from datetime import datetime
now_time = datetime.now().strftime('%Y.%m.%d.%H:%M')
visualize_log('keras-rl/noisynet_pdd_dqn_MsPacman-ram-v0_log.json', output = 'output_model_{}.png'.format(now_time))
visualize_log('keras-rl/noisynet_pdd_dqn_MsPacman-ram-v0_V2_log.json', output = 'output_model_V2_{}.png'.format(now_time))


with open('keras-rl/noisynet_pdd_dqn_MsPacman-ram-v0_log.json', 'r') as f:
        data = json.load(f)

with open('keras-rl/noisynet_pdd_dqn_MsPacman-ram-v0_V2_log.json', 'r') as f:
    data = json.load(f)
episodes = data['episode']

# Get value keys. The x axis is shared and is the number of episodes.
keys = sorted(list(set(data.keys()).difference(set(['episode']))))

import numpy as np
np.mean(data['episode_reward'][-100:])