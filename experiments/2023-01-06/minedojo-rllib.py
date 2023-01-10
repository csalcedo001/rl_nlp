import gym
import ray
from ray.rllib.algorithms import ppo
import numpy as np
import torch
import torch.nn.functional as F
import minedojo
import cv2


class MinedojoRGBWrapper(gym.Env):
    def __init__(self, env_config):
        self.env = env_config['environment']
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space['rgb']

        self.resize_input = True
        if self.resize_input:
            shape = (240, 320, 3)
        else:
            shape = obs_shape[1:] + (obs_shape[0],)

        # if channel is first
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.min(),
            high=self.observation_space.high.max(),
            shape=shape,
        )
    
    def reset(self):
        raw_obs = self.env.reset()

        obs = self.get_obs(raw_obs)
        
        return obs

    def step(self, action):
        action = action.copy()
        action[5] = 0    # Set action to always be no-op

        raw_obs, reward, done, info = self.env.step(action)

        obs = self.get_obs(raw_obs)

        return obs, reward, done, info
    
    def get_obs(self, raw_obs):
        obs = raw_obs['rgb']
        obs = np.transpose(obs, (1, 2, 0))

        if self.resize_input:
            obs = cv2.resize(obs, dsize=self.observation_space.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
            obs = np.clip(obs, 0., 255., dtype=np.float32)

        return obs
        


env = minedojo.make(
    task_id="harvest_wool_with_shears_and_sheep",
    image_size=(160, 256)
)

ray.init()

algo = (
    ppo.PPOConfig()
    .framework(framework='torch')
    .rollouts(num_rollout_workers=1, horizon=1000)
    .resources(num_gpus=1)
    .environment(env=MinedojoRGBWrapper, env_config={
        'environment': env,
        "disable_env_checking": True,
    })
    .build()
)

for _ in range(5):
    print(algo.train())

algo.evaluate()