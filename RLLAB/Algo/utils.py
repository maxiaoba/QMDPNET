import numpy as np
from rllab.misc import tensor_utils
import time


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []

    if hasattr(env._wrapped_env,'generate_grid'):
        env._wrapped_env.generate_grid=False
    if hasattr(env._wrapped_env,'generate_b0_start_goal'):
        env._wrapped_env.generate_b0_start_goal=False
    # print(env._wrapped_env.env_img)
    o = env.reset()
    if hasattr(agent, 'prob_network') and hasattr(agent.prob_network, '_l_gru') and hasattr(agent.prob_network._l_gru, 'map'):
        agent.reset(env._wrapped_env.env_img, env._wrapped_env.goal_img, env._wrapped_env.b0_img)
    else:
        agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        # print('action: ',a)
        # print('env ob: ',o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )
