import gym
import numpy as np

from dreamer.env import RenderObsWrapper


### Print environment shapes as provided by the renderer
env_names = [
    'Acrobot-v1',
    'HalfCheetah-v4',
]

for env_name in env_names:
    env = gym.make(env_name, render_mode='rgb_array')
    env = RenderObsWrapper(env)

    state = env.reset()

    print("ENV: {:>16}, shape: {}".format(env_name, state.shape))