import tensorflow as tf
import joblib
from matplotlib import pyplot
import numpy as np
import os
from Env.grid_env_r_baseline import GridBase
from sandbox.rocky.tf.envs.base import TfEnv

sess = tf.Session()
sess.__enter__()

N=10
M=10
Pmove_succ=1.0
Pobs_succ=1.0

params = {
    'grid_n': N,
    'grid_m': M,
    'K': 30,
    'Pobst': 0.25,  # probability of obstacles in random grid

    'R_obst': -1.0, 'R_goal': 20.0, 'R_step': -0.1,#0.0,#'R_step': -0.1, 'R_obst': -10
    'R_stay': -1.0,
    'discount': 0.99,
    'Pmove_succ':Pmove_succ,
    'Pobs_succ': Pobs_succ,

    'num_action': 5,
    'moves': [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]],  # right, down, left, up, stay
    'stayaction': 4,

    'num_obs': 16,
    'observe_directions': [[0, 1], [1, 0], [0, -1], [-1, 0]],
    }

params['obs_len'] = len(params['observe_directions'])
params['num_state'] = params['grid_n']*params['grid_m']
params['traj_limit'] = 4 * (params['grid_n'] * params['grid_m']) # 4 * (params['grid_n'] + params['grid_m'])
params['R_step'] = [params['R_step']] * params['num_action']
params['R_step'][params['stayaction']] = params['R_stay']
params['kdist'] = -0.1

env = TfEnv(GridBase(params))
env._wrapped_env.generate_grid=True
env._wrapped_env.generate_b0_start_goal=True
env.reset()
env._wrapped_env.generate_grid=False
env._wrapped_env.generate_b0_start_goal=False

log_dir = "../MultiEnv/Data/Baseline"

names = []
for policy in os.listdir(log_dir):
    if not (policy == ".DS_Store"):
        names.append(policy[:-4:])

for name in names:
    tf.reset_default_graph()
    sess.close()
    sess = tf.Session()
    sess.__enter__()
    data = joblib.load(log_dir+'/'+name+'.pkl')
    agent = data['policy']
    max_path_length = 400
    success = 0
    path_lengths = np.array([])

    env_path = "../MultiEnv/"
    for i in range(20):
        data = joblib.load(env_path+'/TestEnv'+'/env_'+str(i)+'.pkl')
        env_ref = data['env']
        grid = env_ref._wrapped_env.grid
        b0 = env_ref._wrapped_env.b0
        start_state = env_ref._wrapped_env.start_state
        goal_state = env_ref._wrapped_env.goal_state
        env._wrapped_env.__init__(env._wrapped_env.params,grid=grid,b0=b0,start_state=start_state,goal_state=goal_state)
        env._wrapped_env.generate_grid=False
        env._wrapped_env.generate_b0_start_goal=False

        o = env.reset()
        agent.reset()
        # agent.reset()
        path_length = 0

        while True:
            a, agent_info = agent.get_action(o)
            next_o, r, d, env_info = env.step(a)
            path_length += 1
            if d:
             	break
            o = next_o

        if path_length < max_path_length:
            success += 1
            path_lengths = np.append(path_lengths,path_length)
    mean_path_length = np.mean(path_lengths)
    print(name)
    print('success: ',success)
    print('mean length: ',mean_path_length)

