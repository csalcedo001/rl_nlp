import gym
import torch
import numpy as np
import cv2
from tqdm import tqdm

from dreamer.env import RenderObsWrapper

env = gym.make('Acrobot-v1', render_mode='rgb_array')
env = RenderObsWrapper(env)


max_iteration = 1000

state = env.reset()

states = []

for iteration in tqdm(range(max_iteration)):
    action = env.action_space.sample()

    next_state, reward, done, _, _ = env.step(action)

    states.append(state)

    if done:
        break

    state = next_state

frame_size = states[0].shape[1:]



print("State shape:", states[0].shape)
print("Frame size:", frame_size)

out = cv2.VideoWriter(
    'output_video.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    20,
    frame_size
)

for state in states:
    frame = np.transpose(state, (1, 2, 0))
    frame = frame.astype(np.uint8)

    out.write(frame)

out.release()