import pathlib as pathlib
import minedojo

# %%
# Initialize Avalon with the default parametric world generator

env = minedojo.make(task_id="harvest_milk", image_size=(160, 256))

print("Task: {}".format(env.task_prompt))
print("Guidance:")
print(env.task_guidance)

task_prompt, task_guidance = minedojo.tasks.ALL_PROGRAMMATIC_TASK_INSTRUCTIONS["harvest_milk"]
obs = env.reset()


# %%
# Take some environment steps and record the observations

def random_env_step():
    action = env.action_space.no_op()
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()
    return obs


observations = [random_env_step() for _ in range(50)]
# display_video(observations, fps=10)
# %%
# We can also generate and load a world manually

# OUTPUT_FOLDER = pathlib.Path("./output/").absolute()

# shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
# params = generate_world(
#     GenerateAvalonWorldParams(
#         AvalonTask.MOVE,
#         difficulty=1,
#         seed=42,
#         index=0,
#         output=str(OUTPUT_FOLDER),
#     )
# )
# env.reset_nicely_with_specific_world(episode_seed=0, world_params=params)

# observations = [random_env_step() for _ in range(50)]
# # %%
# # display_video(observations, fps=10)