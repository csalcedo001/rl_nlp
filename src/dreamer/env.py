import gym
import numpy as np


class RenderObsWrapper(gym.Wrapper):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        self.env = env

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def step(self, action):
        _, reward, done, _, info = self.env.step(action)
        obs = self._get_obs()

        return obs, reward, done, False, info

    def reset(self):
        self.env.reset()

        obs = self._get_obs()

        return obs

    def _get_obs(self):
        return self.env.render().transpose((2, 0, 1))