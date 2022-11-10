import gym
import numpy as np


class RenderObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.env = env

        obs_shape = self.reset().shape
        self._observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=obs_shape
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        obs = self._get_obs()

        return obs, reward, done, info

    def reset(self):
        self.env.reset()

        obs = self._get_obs()

        return obs

    def _get_obs(self):
        return self.env.render()[0]

    def _get_render_obs_shape(self):
        return self.reset().shape