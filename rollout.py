import torch
import numpy as np
import gym

from dreamer import Dreamer


# def get_screen(env):
#     screen = env.render().transpose((2, 0, 1))
#     print("SCREEN SHAPE", screen.shape)
#     _, screen_height, screen_width = screen.shape
#     screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
#     view_width = int(screen_width * 0.6)
#     cart_location = get_cart_location(screen_width)
#     if cart_location < view_width // 2:
#         slice_range = slice(view_width)
#     elif cart_location > (screen_width - view_width // 2):
#         slice_range = slice(-view_width, None)
#     else:
#         slice_range = slice(cart_location - view_width // 2,
#                             cart_location + view_width // 2)
#     # Strip off the edges, so that we have a square image centered on a cart
#     screen = screen[:, :, slice_range]
#     # Convert to float, rescale, convert to torch tensor
#     # (this doesn't require a copy)
#     screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
#     screen = torch.from_numpy(screen)
#     # Resize, and add a batch dimension (BCHW)
#     return resize(screen).unsqueeze(0)

env = gym.make('Acrobot-v1', render_mode='rgb_array')
env.reset()
state = env.render().transpose((2, 0, 1))
print(state.shape)

agent = Dreamer(env, 16)

print(agent)

episodes = 100
max_iteration = 1000

for episode in range(episodes):
    env.reset()
    state =env.render().transpose((2, 0, 1))
    state = get_screen(env)

    for iteration in range(max_iteration):
        print("STATE:", state.shape)
        state = np.transpose(state, (2, 0, 1))
        print("STATE:", state.shape)
        state = torch.from_numpy(state).float()
        state = state.reshape(1, *state.shape)

        action = agent.act(state)

        next_state, reward, done, info = env.step(action)

        print('Episode {}: reward: {}'.format(episode, reward))

        if done:
            break

        state = next_state