import minedojo

env = minedojo.make(
    task_id="harvest_wool_with_shears_and_sheep",
    image_size=(160, 256)
)

print("OBSERVATION SPACE (RGB):", env.observation_space['rgb'])
print("ACTION SPACE:", env.action_space)

print(env.observation_space['rgb'].shape)