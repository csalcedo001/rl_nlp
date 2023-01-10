import os

import minedojo
from tqdm import tqdm
import pickle

from dreamer.utils import save_video


env = minedojo.make(
    task_id="harvest_wool_with_shears_and_sheep",
    image_size=(160, 256)
)
env.reset()

def random_env_step(i):
    action = env.action_space.sample()

    obs, reward, done, info = env.step(action)
    if done:
        env.reset()
    return obs

observations = [random_env_step(i) for i in tqdm(range(200))]

print("OBS.RGB SHAPE:", observations[0]['rgb'].shape)


filename = os.path.join(os.path.dirname(__file__), "minedojo_sample.mp4")

save_video(
    [obs['rgb'] for obs in observations],
    filename,
    channel_first=True, 
    low=0,
    high=255,
    invert_rgb=True,
)