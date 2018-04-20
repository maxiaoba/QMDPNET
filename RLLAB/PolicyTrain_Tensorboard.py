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
import dill
#stub(globals())

# log_dir = "./Data/FixMapStartState"
log_dir = "./Data/TensorBoard"

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


# env = TfEnv(GridBase())
# env._wrapped_env.generate_grid=True
# env._wrapped_env.generate_b0_start_goal=True
# env.reset()
# env._wrapped_env.generate_grid=False
# env._wrapped_env.generate_b0_start_goal=False

# params = dict(
#     env=env,
# )
# joblib.dump(params,log_dir+'/env.pkl')

params = joblib.load('./env.pkl')
env = params['env']

policy = QMDPPolicy(
    env_spec=env.spec,
    name="QMDP",
    qmdp_param=env._wrapped_env.params
)


baseline = LinearFeatureBaseline(env_spec=env.spec)

with tf.Session() as sess:

    writer = tf.summary.FileWriter(logdir=log_dir,)

    algo = VPG_t(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=2*env._wrapped_env.params['traj_limit'],
        max_path_length=env._wrapped_env.params['traj_limit'],
        n_itr=1,
        discount=0.95,
        step_size=0.01,
        record_rewards=True,
        transfer=False,
        # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
        # optimizer = PenaltyLbfgsOptimizer()
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )

    algo.train(sess)
    # tf.summary.merge_all()
    # print(sess.graph)
    writer.add_graph(sess.graph)
    writer.close()