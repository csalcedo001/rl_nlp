import os

import minedojo
from tqdm import tqdm

from dreamer.utils import save_video
from utils import random_env_step, print_space


epochs = 200


env = minedojo.make(
    task_id="harvest_wool_with_shears_and_sheep",
    image_size=(160, 256)
)
env.reset()


print('**************** OBSERVATION SPACE ****************')
print_space(env.observation_space, framework='minedojo')
print('**************** OBSERVATION SPACE ****************\n')

print('**************** ACTION SPACE ****************')
print_space(env.action_space, framework='minedojo')
print('**************** ACTION SPACE ****************\n')


observations = [random_env_step(i, env, framework='minedojo') for i in tqdm(range(epochs))]

filename = os.path.join(os.path.dirname(__file__), "minedojo_video.mp4")

save_video(
    [obs['rgb'] for obs in observations],
    filename,
    channel_first=True, 
    low=0,
    high=255,
    invert_rgb=True,
)