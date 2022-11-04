import gym
import torch
import numpy as np
from tqdm import tqdm

from dreamer.env import RenderObsWrapper

env = gym.make('Acrobot-v1', render_mode='rgb_array')
env = RenderObsWrapper(env)

episodes = 10
max_iteration = 100

print("Starting rollouts...")
for episode in range(episodes):
    state = env.reset()

    total_reward = 0.

    for iteration in tqdm(range(max_iteration)):
        state = torch.from_numpy(state).float()
        state = state.reshape(1, *state.shape)

        action = env.action_space.sample()

        next_state, reward, done, _, info = env.step(action)
        total_reward += reward

        if done:
            break

        state = next_state

    print('Episode: {}, reward: {}'.format(episode, reward))