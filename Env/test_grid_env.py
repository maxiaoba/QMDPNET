from grid_env_r import GridBase
import numpy as np
from sandbox.rocky.tf.envs.base import TfEnv

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

env = GridBase(params)
env.generate_grid=True
env.generate_b0_start_goal=True
env.reset()
tfenv = TfEnv(env)
tfenv.step(1)