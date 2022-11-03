import gym
import numpy as np

from dreamer.env import RenderObsWrapper

env_name = 'Acrobot-v1'

env = gym.make(env_name, render_mode='rgb_array')
env = RenderObsWrapper(env)

state = env.reset()

print("ENV: {:>16}, shape: {}".format(env_name, state.shape))