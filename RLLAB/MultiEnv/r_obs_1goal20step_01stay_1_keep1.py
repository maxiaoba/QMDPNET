import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

from Algo.vpg_transfer_multimaze import VPG_t
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

env_path = "./TrainEnv"
log_dir = "./Data/obs_1goal20step_01stay_1_keep1"

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

with tf.Session() as sess:
    params = joblib.load(log_dir+'/params.pkl')
    itr=params['itr']
    policy=params['policy']
    baseline=params['baseline']
    env=params['env']
    rewards=params['rewards']

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
        transfer=True,
        env_path=env_path,
        env_num=500,
        env_keep_itr=1,
        rewards=rewards,
        start_itr=itr,
    )

    algo.train(sess)