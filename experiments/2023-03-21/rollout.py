import os
import gym
import time
import argparse
from tqdm import tqdm
import wandb
from dotenv import load_dotenv

load_dotenv()

from utils import random_env_step, get_frames_from_observations, print_space, save_video



if 'WANDB_ENTITY' not in os.environ:
    raise ValueError('WANDB_ENTITY not found in environment variables')
WANDB_ENTITY = os.environ.get('WANDB_ENTITY')

if 'WANDB_PROJECT' not in os.environ:
    raise ValueError('WANDB_PROJECT not found in environment variables')
WANDB_PROJECT = os.environ.get('WANDB_PROJECT')



parser = argparse.ArgumentParser()
parser.add_argument('--framework', required=True, type=str, choices=['minerl', 'minedojo'])
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument('--episodes', type=int, default=100)

args = parser.parse_args()



# Init W&B
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    config=args,
)

### Test total step time
time_data = {
    'total': None,
    'import': None,
    'gym_make': None,
    'first_reset': None,
    'average_reset': None,
    'average_step': None,
    'reset': [],
    'step': [],
}
total_start_time = time.time()

### Test package import time
start_time = time.time()
if args.framework == 'minerl':
    import minerl
elif args.framework == 'minedojo':
    import minedojo
end_time = time.time()
time_data['import'] = end_time - start_time

wandb.log({
    'time': {
        'import': time_data['import'],
    }
})



### Test environment creation time
start_time = time.time()
if args.framework == 'minerl':
    env = gym.make('MineRLObtainDiamond-v0') # MineRL uses 64x64 by default
elif args.framework == 'minedojo':
    env = minedojo.make(
        task_id="harvest_1_diamond",
        image_size=(64, 64)
)
end_time = time.time()
time_data['gym_make'] = end_time - start_time

wandb.log({
    'time': {
        'gym_make': time_data['gym_make'],
    }
})



### Print observation space and action space
print('**************** OBSERVATION SPACE ****************')
print_space(env.observation_space, framework=args.framework)
print('**************** OBSERVATION SPACE ****************\n')

print('**************** ACTION SPACE ****************')
print_space(env.action_space, framework=args.framework)
print('**************** ACTION SPACE ****************\n')



### Test environment reset time
start_time = time.time()
env.reset()
end_time = time.time()
time_data['first_reset'] = end_time - start_time

wandb.log({
    'time': {
        'first_reset': time_data['first_reset'],
    }
})



for episode in tqdm(range(args.episodes)):
    ### Test environment reset time
    start_time = time.time()
    env.reset()
    end_time = time.time()
    time_data['reset'].append(end_time - start_time)



    ### Test step time
    start_time = time.time()
    observations = [random_env_step(i, env, framework=args.framework) for i in tqdm(range(args.iterations))]
    end_time = time.time()
    time_data['step'].append((end_time - start_time) / args.iterations)

    wandb.log({
        'time': {
            'reset': time_data['reset'][-1],
            'step': time_data['step'][-1],
        }
    })

time_data['average_reset'] = sum(time_data['reset']) / len(time_data['reset'])
time_data['average_step'] = sum(time_data['step']) / len(time_data['step'])

wandb.log({
    'time': {
        'average_reset': time_data['average_reset'],
        'average_step': time_data['average_step'],
    }
})



### Conclude main step time
total_end_time = time.time()
time_data['total'] = total_end_time - total_start_time


### Save video
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

### Print step time results
print('**************** TIME DATA ****************')
for key, value in time_data.items():
    if isinstance(value, str):
        print(f"{key}: {value:.5f}s")
    else:
        print(f"{key}: {value}")

wandb.log({
    'time': {
        'total': time_data['total'],
    }
})
