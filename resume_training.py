from sandbox.rocky.tf.algos.trpo import TRPO
from Algo.trpo_transfer import TRPO_t
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

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

import uuid
filename = str(uuid.uuid4())
###
# def segment_handler(signum, frame): 
#     print("###################resume training#######################")
#     command = ["python scripts/resume_training_tf.py "]
#     subprocess.call(command, shell=True, env=os.environ)
###
if __name__ == "__main__":
    ###
    #signal.signal(signal.SIGCHLD, segment_handler)
    log_dir = "./Data/FixMapStartState"

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
    logger.set_snapshot_gap(10)
    logger.set_log_tabular_only(False)
    logger.push_prefix("[%s] " % "FixMapStartState")

    from Algo import parallel_sampler
    parallel_sampler.initialize(n_parallel=1)
    parallel_sampler.set_seed(0)


    with tf.Session() as sess:
        # file = open(pkl_file, 'rb')
        # data = dill.load(file)
        data = joblib.load(pkl_file)
        env = data['env']
        idx = data['itr']
        policy = data['policy']
        baseline = data['baseline']

        rewards = None
        if 'rewards' in data.keys():
            print("already has rewards~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            rewards = data['rewards']
        algo = TRPO_t(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=2048,
            max_path_length=400,
            n_itr=5000,
            discount=0.95,
            step_size=0.01,
            record_rewards=True,
            transfer=True,
            optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
            start_itr=idx,
            # Uncomment both lines (this and the plot parameter below) to enable plotting
            # plot=True,
        )
        algo.train(sess)
        #assert 'algo' in data
        #algo = data['algo']
        #algo.train()
