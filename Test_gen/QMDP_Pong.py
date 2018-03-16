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

from Env.atari import AtariEnv
from rllab.misc.instrument import stub, run_experiment_lite
from Policy_gen.qmdp_policy import QMDPPolicy
from sandbox.rocky.tf.policies.categorical_gru_policy import CategoricalGRUPolicy

import lasagne.nonlinearities as NL
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc import logger
import os.path as osp
import tensorflow as tf
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
import joblib
import dill



env = TfEnv(AtariEnv(5,'carnival'))
qmdp_param = {}
qmdp_param['K'] = 30
qmdp_param['obs_len'] = env.spec.observation_space.flat_dim
qmdp_param['num_action'] = env.spec.action_space.flat_dim
qmdp_param['num_state'] = 32 #env.spec.observation_space.flat_dim
qmdp_param['info_len'] = qmdp_param['num_state']

# log_dir = "./Data/FixMapStartState"
log_dir = "./Data/testcarnival"

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

policy = QMDPPolicy(
    env_spec=env.spec,
    name="QMDP",
    qmdp_param=qmdp_param,
)


baseline = LinearFeatureBaseline(env_spec=env.spec)

with tf.Session() as sess:

    # writer = tf.summary.FileWriter(logdir=log_dir,)

    algo = VPG_t(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=2048,#2*env._wrapped_env.params['traj_limit'],
        max_path_length=200,
        n_itr=10000,
        discount=0.95,
        step_size=0.01,
        record_rewards=True,
        transfer=False,
    )

    algo.train(sess)
    # tf.summary.merge_all()
    # print(sess.graph)
    # writer.add_graph(sess.graph)
    # writer.close()