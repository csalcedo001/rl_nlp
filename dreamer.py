import os

os.environ['MUJOCO_GL'] = 'egl'

import torch
from torch import nn
from torch import distributions

class Dreamer(nn.Module):
    def __init__(self, env, hidden):
        super().__init__()

        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape

        channels = obs_shape[2]

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, 6, 2),
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

    def act(self, obs):
        batch_size = obs.shape[0]
        print("OBS", obs.shape)

        z = self.encoder(obs)
        print("Z", z.shape)
        obs_hat = self.decoder(z.reshape(batch_size, -1, 1, 1))


        # if self.train:
        #     self.encoder()

        return action