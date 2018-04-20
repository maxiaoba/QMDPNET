import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.vpg import VPG
from Algo.trpo_transfer import TRPO_t
from Algo.vpg_transfer import VPG_t
from Algo.npo_transfer import NPO_t
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer

from Env.grid_env import GridBase
from rllab.misc.instrument import stub, run_experiment_lite
from qmdp_policy import QMDPPolicy
from sandbox.rocky.tf.policies.categorical_gru_policy import CategoricalGRUPolicy

import lasagne.nonlinearities as NL
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc import logger
import os.path as osp
import tensorflow as tf
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
import joblib

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

env_ref = joblib.load('./env.pkl')['env']
grid = env_ref._wrapped_env.grid
b0 = env_ref._wrapped_env.b0
start_state = env_ref._wrapped_env.start_state
goal_state = env_ref._wrapped_env.goal_state
env = TfEnv(GridBase(params,grid=grid,b0=b0,start_state=start_state,goal_state=goal_state))
env._wrapped_env.generate_grid=False
env._wrapped_env.generate_b0_start_goal=False
env.reset()

log_dir = "./Data/obs_1goal20step_01stay_1_gru"

tabular_log_file = osp.join(log_dir, "progress.csv")
text_log_file = osp.join(log_dir, "debug.log")
params_log_file = osp.join(log_dir, "params.json")
pkl_file = osp.join(log_dir, "params.pkl")

logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
prev_snapshot_dir = logger.get_snapshot_dir()
prev_mode = logger.get_snapshot_mode()
logger.set_snapshot_dir(log_dir)
logger.set_snapshot_mode("gaplast")
logger.set_snapshot_gap(1000)
logger.set_log_tabular_only(False)
logger.push_prefix("[%s] " % "FixMapStartState")

from Algo import parallel_sampler
parallel_sampler.initialize(n_parallel=1)
parallel_sampler.set_seed(0)

policy = CategoricalGRUPolicy(
    env_spec=env.spec,
    name="gru",
)


baseline = LinearFeatureBaseline(env_spec=env.spec)

with tf.Session() as sess:
    writer = tf.summary.FileWriter(logdir=log_dir,)

    algo = VPG_t(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=2048,#2*env._wrapped_env.params['traj_limit'],
        max_path_length=env._wrapped_env.params['traj_limit'],
        n_itr=10000,
        discount=0.95,
        step_size=0.01,
        record_rewards=True,
        transfer=False,
    )
    algo.train(sess)
    writer.add_graph(sess.graph)
    writer.close()