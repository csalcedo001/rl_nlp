import gym
import torch
import numpy as np

from dreamer import Dreamer
from dreamer.env import RenderObsEnv

env = gym.make('Acrobot-v1', render_mode='rgb_array')
env = RenderObsEnv(env)

episodes = 10
max_iteration = 1000

for episode in range(episodes):
    state = env.reset()

    for iteration in range(max_iteration):
        state = torch.from_numpy(state).float()
        state = state.reshape(1, *state.shape)

        action = env.action_space.sample()

        next_state, reward, done, _, info = env.step(action)

        print('Episode {}: reward: {}'.format(episode, reward))

        if done:
            break

        state = next_state