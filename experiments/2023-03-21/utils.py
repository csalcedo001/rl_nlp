def random_env_step(env, framework):
    action = env.action_space.no_op()

    if framework == "minerl":
        action["ESC"] = 0 # to avoid ending the episode early

    obs, _, done, _ = env.step(action)
    if done:
        env.reset()
    
    return obs

def print_observation_space_dict(env, framework):
    dict_cls = _get_dict_class_from_framework(framework)
    
    print('**************** OBSERVATION SPACE ****************')
    _print_dict_r(env.observation_space, dict_cls)
    print('**************** OBSERVATION SPACE ****************\n')

def print_action_space_dict(env, framework):
    dict_cls = _get_dict_class_from_framework(framework)
    
    print('**************** ACTION SPACE ****************')
    _print_dict_r(env.action_space, dict_cls)
    print('**************** ACTION SPACE ****************\n')


def _print_dict_r(obj, dict_cls, key=None, tabs=0):
    offset = ' ' * tabs * 4

    if key == None:
        print(offset, end='')
    else:
        print(offset, key, ': ', sep='', end='')

    if not isinstance(obj, dict_cls):
        print(obj, ',', sep='')
        return

    print('{', sep='')

    for key in obj:
        value = obj[key]

        _print_dict_r(value, dict_cls, key=key, tabs=tabs + 1)

    print(offset, '},', sep='')

def _get_dict_class_from_framework(framework):
    if framework == 'minerl':
        import minerl
        dict_cls = minerl.herobraine.hero.spaces.Dict
    elif framework == 'minedojo':
        import gym
        dict_cls = gym.spaces.Dict
    else:
        raise ValueError(f"Invalid framework: {framework}")
    
    return dict_cls