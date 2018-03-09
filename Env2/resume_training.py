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

log_dir = "./Data/obs_1goal20step0stay_1"

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

    # initialize uninitialize variables
    # global_vars          = tf.global_variables()
    # print([str(v.name) for v in global_vars])
    # is_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    # not_initialized_vars = [v for (v, f) in zip(global_vars, is_initialized) if not f]

    # print([str(i.name) for i in not_initialized_vars]) # only for testing

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
        rewards=rewards,
        transfer=True,
        start_itr=itr,
    )

    # algo.train(sess)