import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

from Algo.vpg_transfer_multimaze import VPG_t
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer

from Env.grid_env_baseline import GridBase
from rllab.misc.instrument import stub, run_experiment_lite
from qmdp_policy import QMDPPolicy
from sandbox.rocky.tf.policies.categorical_conv_policy import CategoricalConvPolicy

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

env = TfEnv(GridBase(params))
env._wrapped_env.generate_grid=True
env._wrapped_env.generate_b0_start_goal=True
env.reset()
env._wrapped_env.generate_grid=False
env._wrapped_env.generate_b0_start_goal=False
# log_dir = "./Data/FixMapStartState"
env_path = "./TrainEnv"
log_dir = "./Data/b_obs_1goal20step_01stay_1_keep3"

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
logger.set_snapshot_gap(100)
logger.set_log_tabular_only(False)
logger.push_prefix("[%s] " % "FixMapStartState")

from Algo import parallel_sampler
parallel_sampler.initialize(n_parallel=1)
parallel_sampler.set_seed(0)

policy = CategoricalConvPolicy(
    env_spec=env.spec,
    name="ConvNet",
    conv_filters=[3,3,3,3], 
    conv_filter_sizes=[3,5,3,3], 
    conv_strides=[1,1,1,1], 
    conv_pads=['SAME','SAME','SAME','SAME'],
)


baseline = LinearFeatureBaseline(env_spec=env.spec)

with tf.Session() as sess:

    algo = VPG_t(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=2048,
        max_path_length=env._wrapped_env.params['traj_limit'],
        n_itr=20000,
        discount=0.95,
        step_size=0.01,
        record_rewards=True,
        transfer=False,
        env_path=env_path,
        env_num=500,
        env_keep_itr=3,
    )

    algo.train(sess)