import time
from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
from sandbox.rocky.tf.policies.base import Policy
import tensorflow as tf
from Algo.qmdp_sampler import QMDPSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from rllab.sampler.utils import rollout

from sandbox.rocky.tf.algos.npo import NPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.misc.overrides import overrides
import numpy as np

class TRPO_t(NPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            transfer=True,
            optimizer=None,
            optimizer_args=None,
            record_env=True
            **kwargs):
        self.transfer = transfer
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)

        self.record_env = record_env
        super(TRPO_t, self).__init__(optimizer=optimizer, sampler_cls=QMDPSampler,sampler_args=dict(),**kwargs)

    @overrides
    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()
        if not self.transfer:
            sess.run(tf.global_variables_initializer())
        self.start_worker()
        start_time = time.time()
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Obtaining samples...")
                # self.env._wrapped_env.generate_grid=True
                # self.env._wrapped_env.generate_b0_start_goal=True
                # self.env.reset()
                # self.env._wrapped_env.generate_grid=False
                # self.env._wrapped_env.generate_b0_start_goal=False
                paths = self.obtain_samples(itr)
                logger.log("Processing samples...")
                samples_data = self.process_samples(itr, paths)

                logger.log("Logging diagnostics...")
                self.log_diagnostics(paths)
                logger.log("Optimizing policy...")
                self.optimize_policy(itr, samples_data)
                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr)  # , **kwargs)
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    rollout(self.env, self.policy, animated=True, max_path_length=self.max_path_length)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")
        self.shutdown_worker()
        if created_session:
            sess.close()

    @overrides
    def get_itr_snapshot(self, itr):
        if self.record_env:
            return dict(
                itr=itr,
                policy=self.policy,
                baseline=self.baseline,
                env=self.env,
            )
        else:
            return dict(
                itr=itr,
                policy=self.policy,
                baseline=self.baseline,
            )
