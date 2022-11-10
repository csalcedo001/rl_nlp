import gym
import torch
import numpy as np
from tqdm import tqdm

from dreamer.env import RenderObsWrapper
from dreamer.utils import save_video

env = gym.make('HalfCheetah-v4', render_mode='rgb_array')
env = RenderObsWrapper(env)


max_iteration = 1000

state = env.reset()

states = []

for iteration in tqdm(range(max_iteration)):
    action = env.action_space.sample()

    next_state, reward, done, _ = env.step(action)

    states.append(state)

    if done:
        break

    state = next_state



save_video(states, 'videos/last.mp4')