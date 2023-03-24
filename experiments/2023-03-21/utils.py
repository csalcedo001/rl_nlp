def random_env_step(env, framework):
    action = env.action_space.no_op()

    if framework == "minerl":
        action["ESC"] = 0 # to avoid ending the episode early

    obs, _, done, _ = env.step(action)
    if done:
        env.reset()
    
    return obs

def print_observation_space_dict(env, framework):
    leaves = _get_print_leaves_from_framework(framework)
    
    print('**************** OBSERVATION SPACE ****************')
    _print_dict_r(env.observation_space, leaves)
    print('**************** OBSERVATION SPACE ****************\n')

def print_action_space_dict(env, framework):
    leaves = _get_print_leaves_from_framework(framework)
    
    print('**************** ACTION SPACE ****************')
    _print_dict_r(env.action_space, leaves)
    print('**************** ACTION SPACE ****************\n')


def _print_dict_r(obj, leaves, key=None, tabs=0):
    offset = ' ' * tabs * 4

    if key == None:
        print(offset, end='')
    else:
        print(offset, key, ': ', sep='', end='')

    is_leaf_node = sum([isinstance(obj, leaf) for leaf in leaves]) > 0
    if is_leaf_node:
        print(obj, ',', sep='')
        return

    print('{', sep='')

    for key in obj:
        value = obj[key]

        _print_dict_r(value, leaves, key=key, tabs=tabs + 1)

    print(offset, '},', sep='')

def _get_print_leaves_from_framework(framework):
    if framework == 'minerl':
        import minerl
        leaves = [
            minerl.herobraine.hero.spaces.Box,
            minerl.herobraine.hero.spaces.Discrete,
        ]
    elif framework == 'minedojo':
        import gym
        leaves = [
            gym.spaces.box.Box,
            gym.spaces.box.Discrete,
        ]
    else:
        raise ValueError(f"Invalid framework: {framework}")
    
    return leaves