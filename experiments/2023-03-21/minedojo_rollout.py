import os
import time
import argparse
from tqdm import tqdm

from dreamer.utils import save_video
from utils import random_env_step, print_space



parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=200)
args = parser.parse_args()



### Test overall execution time
time_data = {}
main_start_time = time.time()

### Test package import time
start_time = time.time()
import minedojo
end_time = time.time()
time_data['import'] = end_time - start_time



### Test environment creation time
start_time = time.time()
env = minedojo.make(
    task_id="harvest_wool_with_shears_and_sheep",
    image_size=(160, 256)
)
end_time = time.time()
time_data['gym_make'] = end_time - start_time

env.reset()


print('**************** OBSERVATION SPACE ****************')
print_space(env.observation_space, framework='minedojo')
print('**************** OBSERVATION SPACE ****************\n')

print('**************** ACTION SPACE ****************')
print_space(env.action_space, framework='minedojo')
print('**************** ACTION SPACE ****************\n')


### Test execution time
start_time = time.time()
observations = [random_env_step(i, env, framework='minedojo') for i in tqdm(range(args.episodes))]
end_time = time.time()
time_data['execution'] = end_time - start_time

### Conclude main execution time
main_end_time = time.time()
time_data['main'] = main_end_time - main_start_time



filename = os.path.join(os.path.dirname(__file__), "minedojo_video.mp4")

save_video(
    [obs['rgb'] for obs in observations],
    filename,
    channel_first=True, 
    low=0,
    high=255,
    invert_rgb=True,
)

### Print execution time results
print('**************** TIME DATA ****************')
for key, value in time_data.items():
    print(f"{key}: {value:.5f}s")