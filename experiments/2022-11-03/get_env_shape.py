import gym
import torch
from torch import nn
import numpy as np
from torchvision.transforms import Compose, CenterCrop, Resize, ToPILImage, ToTensor
from tqdm import tqdm

from dreamer.env import RenderObsWrapper
from dreamer.utils import save_video


def get_render_obs_shape(env):
    _ = env.reset()

    obs = env.render()


### Print environment shapes as provided by the renderer
env_names = [
    'Acrobot-v1',
    'HalfCheetah-v4',
]

max_iteration = 200

for env_name in env_names:
    env = gym.make(env_name, render_mode='rgb_array')
    env = RenderObsWrapper(env)

    print(env.observation_space.shape)

    transform = Compose([
        ToPILImage(),
        Resize(64),
        ToTensor(),
    ])

    state = env.reset()

    print("ENV: {:>16}, shape: {}".format(env_name, state.shape))

    states = []

    for _ in tqdm(range(max_iteration)):
        action = env.action_space.sample()

        next_state, reward, done, _, _ = env.step(action)

        state = transform(state)
        states.append(state.numpy())

        if done:
            break

        state = next_state


    filename = 'experiments/2022-11-03/{}.mp4'.format(env_name)
    save_video(states, filename)