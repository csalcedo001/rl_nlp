import os

os.environ['MUJOCO_GL'] = 'egl'

import torch
from torch import nn
from torch import distributions

class Dreamer(nn.Module):
    def __init__(self,
            env,
            hidden=32,
            latent_size=100,
        ):
        super().__init__()

        self.env = env

        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape

        channels = obs_shape[2]

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden * 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden * 2, hidden * 4, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden * 4, hidden * 8, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden * 8, latent_size, 4, 1, 1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_size, hidden * 8, 4, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden * 8, hidden * 4, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden * 4, hidden * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden * 2, hidden, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, channels, 4, 2, 1),
            nn.ReLU(),
        )

        # def encoder(obs):
        #     hidden = tf.reshape(obs['image'], [-1] + obs['image'].shape[2:].as_list())
        #     hidden = tf.layers.conv2d(hidden, 32, 4, **kwargs)
        #     hidden = tf.layers.conv2d(hidden, 64, 4, **kwargs)
        #     hidden = tf.layers.conv2d(hidden, 128, 4, **kwargs)
        #     hidden = tf.layers.conv2d(hidden, 256, 4, **kwargs)
        #     hidden = tf.layers.flatten(hidden)
        #     assert hidden.shape[1:].as_list() == [1024], hidden.shape.as_list()
        #     hidden = tf.reshape(hidden, tools.shape(obs['image'])[:2] + [
        #         np.prod(hidden.shape[1:].as_list())])
        #     return hidden


        # def decoder(features, data_shape, std=1.0):
        #     kwargs = dict(strides=2, activation=tf.nn.relu)
        #     hidden = tf.layers.dense(features, 1024, None)
        #     hidden = tf.reshape(hidden, [-1, 1, 1, hidden.shape[-1].value])
        #     hidden = tf.layers.conv2d_transpose(hidden, 128, 5, **kwargs)
        #     hidden = tf.layers.conv2d_transpose(hidden, 64, 5, **kwargs)
        #     hidden = tf.layers.conv2d_transpose(hidden, 32, 6, **kwargs)
        #     mean = tf.layers.conv2d_transpose(hidden, data_shape[-1], 6, strides=2)
        #     assert mean.shape[1:].as_list() == data_shape, mean.shape
        #     mean = tf.reshape(mean, tools.shape(features)[:-1] + data_shape)
        #     return tfd.Independent(tfd.Normal(mean, std), len(data_shape))

    def act(self, z):
        return self.env.action_space.sample()

    def encode(self, obs):
        z = self.encoder(obs)

        return z
    
    def decode(self, z):
        obs_hat = self.decoder(z)

        return obs_hat