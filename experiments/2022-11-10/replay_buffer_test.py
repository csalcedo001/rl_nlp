import gym
import torch
import numpy as np
from tqdm import tqdm

from dreamer.env import RenderObsWrapper
from dreamer.replay_buffer import ReplayBuffer

env = gym.make('Acrobot-v1', render_mode='rgb_array')
env = RenderObsWrapper(env)

capacity = 1024
replay_buffer = ReplayBuffer(capacity)

max_iteration = 100

while replay_buffer.size < replay_buffer.capacity:
    print("Replay buffer. Size: {}, Capacity: {}.".format(replay_buffer.size, replay_buffer.capacity))

    state = env.reset()

    for iteration in tqdm(range(max_iteration)):
        action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)

        replay_buffer.store(state, action, reward, next_state)

        if done:
            break

        state = next_state

print("Replay buffer. Size: {}, Capacity: {}.".format(replay_buffer.size, replay_buffer.capacity))

batch_size = 16
samples = replay_buffer.sample(batch_size)


print("\n\nSampling {} (s,a,r,s) tuples from replay buffer.".format(batch_size))

for i, sample in enumerate(samples):
    state, action, reward, next_state = sample

    print("– Sample {:>2}. s.shape: {}, a: {}, next_s.shape: {}, reward:, {}".format(
        i + 1,
        state.shape,
        action,
        next_state.shape,
        reward,
    ))