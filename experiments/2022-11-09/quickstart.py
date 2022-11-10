from avalon.agent.godot.godot_gym import GodotEnvironmentParams
from avalon.agent.godot.godot_gym import TrainingProtocolChoice
from avalon.agent.godot.godot_gym import AvalonEnv
from avalon.datagen.env_helper import display_video
from dreamer.utils import save_video
from tqdm import tqdm

env_params = GodotEnvironmentParams(
    resolution=256,
    training_protocol=TrainingProtocolChoice.SINGLE_TASK_FIGHT,
    initial_difficulty=1,
)
env = AvalonEnv(env_params)
env.reset()

def random_env_step():
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()
    return obs

observations = [random_env_step() for _ in tqdm(range(200))]
save_video(
    [obs['rgbd'][:, :, :3] for obs in observations],
    "experiments/2022-11-09/avalon_sample.mp4",
    channel_first=False, 
    low=0,
    high=255
)
# display_video(observations, fps=10)
