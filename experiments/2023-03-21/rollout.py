import os
import gym
import time
import argparse
from tqdm import tqdm

from dreamer.utils import save_video
from utils import random_env_step, get_frames_from_observations, print_space



parser = argparse.ArgumentParser()
parser.add_argument('--framework', required=True, type=str, choices=['minerl', 'minedojo'])
parser.add_argument('--episodes', type=int, default=200)

args = parser.parse_args()



### Test overall execution time
time_data = {}
main_start_time = time.time()

### Test package import time
start_time = time.time()
if args.framework == 'minerl':
    import minerl
elif args.framework == 'minedojo':
    import minedojo
end_time = time.time()
time_data['import'] = end_time - start_time



### Test environment creation time
start_time = time.time()
if args.framework == 'minerl':
    env = gym.make('MineRLNavigateDense-v0')
elif args.framework == 'minedojo':
    env = minedojo.make(
        task_id="harvest_wool_with_shears_and_sheep",
        image_size=(160, 256)
)
end_time = time.time()
time_data['gym_make'] = end_time - start_time

env.reset()


print('**************** OBSERVATION SPACE ****************')
print_space(env.observation_space, framework=args.framework)
print('**************** OBSERVATION SPACE ****************\n')

print('**************** ACTION SPACE ****************')
print_space(env.action_space, framework=args.framework)
print('**************** ACTION SPACE ****************\n')


### Test execution time
start_time = time.time()
observations = [random_env_step(i, env, framework=args.framework) for i in tqdm(range(args.episodes))]
end_time = time.time()
time_data['execution'] = end_time - start_time

### Conclude main execution time
main_end_time = time.time()
time_data['main'] = main_end_time - main_start_time


video_name = f'{args.framework}_video.mp4'
filename = os.path.join(os.path.dirname(__file__), video_name)

frames = get_frames_from_observations(observations, args.framework)
save_video(
    frames,
    filename,
    channel_first=args.framework == 'minedojo', 
    low=0,
    high=255,
    invert_rgb=True,
)

### Print execution time results
print('**************** TIME DATA ****************')
for key, value in time_data.items():
    print(f"{key}: {value:.5f}s")