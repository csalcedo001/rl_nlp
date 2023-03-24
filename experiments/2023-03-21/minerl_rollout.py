import os

import gym
import minerl
from tqdm import tqdm

from dreamer.utils import save_video
from utils import random_env_step, print_space


epochs = 200


env = gym.make('MineRLNavigateDense-v0')
obs = env.reset()


print('**************** OBSERVATION SPACE ****************')
print_space(env.observation_space, framework='minerl')
print('**************** OBSERVATION SPACE ****************\n')


print('**************** ACTION SPACE ****************')
print_space(env.action_space, framework='minerl')
print('**************** ACTION SPACE ****************\n')


observations = [random_env_step(i, env, framework='minerl') for i in tqdm(range(epochs))]


filename = os.path.join(os.path.dirname(__file__), "minerl_video.mp4")

save_video(
    [obs['pov'] for obs in observations],
    filename,
    channel_first=False, 
    low=0,
    high=255,
    invert_rgb=True,
)