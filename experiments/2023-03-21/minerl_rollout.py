import os

import gym
import minerl
from tqdm import tqdm

from dreamer.utils import save_video
from utils import random_env_step, print_observation_space_dict, print_action_space_dict


epochs = 10


env = gym.make('MineRLNavigateDense-v0')
obs = env.reset()


observations = [random_env_step(env, framework='minerl') for _ in tqdm(range(epochs))]

print_observation_space_dict(env, framework='minerl')
print_action_space_dict(env, framework='minerl')

# observations = [random_env_step(i) for i in tqdm(range(200))]

# done = False

# for epoch in range(epochs):
#     while True:
#         # Take a random action
#         action = env.action_space.sample()
#         # In BASALT environments, sending ESC action will end the episode
#         # Lets not do that
#         action["ESC"] = 0
#         obs, reward, done, _ = env.step(action)
#         env.render()

#         if done:
#             break


# filename = os.path.join(os.path.dirname(__file__), "minedojo_sample.mp4")

# save_video(
#     [obs['rgb'] for obs in observations],
#     filename,
#     channel_first=True, 
#     low=0,
#     high=255,
#     invert_rgb=True,
# )