import gym
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from dreamer import Dreamer
from dreamer.env import RenderObsWrapper
from dreamer.utils import save_video


env_name = 'Acrobot-v1'

env = gym.make(env_name, render_mode='rgb_array')
env = RenderObsWrapper(env)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(64),
    transforms.ToTensor(),
])

state = env.reset()
print("ENV: {:>16}, shape: {}".format(env_name, state.shape))

agent = Dreamer(env, 16)

print(agent)

episodes = 1
max_iteration = 100

obs_list = []
obs_hat_list = []

with torch.no_grad():
    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0.

        for iteration in tqdm(range(max_iteration)):
            obs = transform(obs)
            z = agent.encode(obs)
            obs_hat = agent.decode(z)
            
            action = agent.act(z)

            next_obs, reward, done, _, info = env.step(action)

            total_reward += reward
            obs_list.append(obs.numpy())
            obs_hat_list.append(obs_hat.numpy())

            if done:
                break

            obs = next_obs
        
        print('Episode [{:>3}/{:>3}]: reward: {}'.format(episode + 1, episodes, total_reward))

filename = 'videos/rollout_ae_input_last.mp4'
save_video(obs_list, filename)

filename = 'videos/rollout_ae_output_last.mp4'
save_video(obs_hat_list, filename)