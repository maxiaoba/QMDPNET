import time
from rllab.algos.base import RLAlgorithm
from sandbox.rocky.tf.policies.base import Policy
from Algo.qmdp_sampler import QMDPSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
import numpy as np

from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf


class NPO_t(BatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            transfer=True,
            record_rewards=True,
            rewards=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(name='PenaltyLbfgsOptimizer')
        self.optimizer = optimizer
        self.step_size = step_size

        self.transfer = transfer
        self.record_rewards = record_rewards
        if self.record_rewards:
            if rewards is None: #create empty dict
                self.rewards = {}
                self.rewards['average_discounted_return'] = []
                self.rewards['AverageReturn'] = []
                self.rewards['StdReturn'] = []
                self.rewards['MaxReturn'] = []
                self.rewards['MinReturn'] = []
            else:
                self.rewards = rewards
        super(NPO_t, self).__init__(env=env, policy=policy, baseline=baseline, sampler_cls=QMDPSampler,sampler_args=dict(), **kwargs)

    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = tensor_utils.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )
        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if is_recurrent:
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            surr_loss = - tf.reduce_sum(lr * advantage_var * valid_var) / tf.reduce_sum(valid_var)
        else:
            mean_kl = tf.reduce_mean(kl)
            surr_loss = - tf.reduce_mean(lr * advantage_var)

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(all_input_values)
        logger.log("Computing KL before")
        mean_kl_before = self.optimizer.constraint_val(all_input_values)
        logger.log("Optimizing")
        self.optimizer.optimize(all_input_values)
        logger.log("Computing KL after")
        mean_kl = self.optimizer.constraint_val(all_input_values)
        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

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

                if self.record_rewards:
                    logger.log("recording rewards...")
                    undiscounted_returns = [sum(path["rewards"]) for path in paths]
                    average_discounted_return = np.mean([path["returns"][0] for path in paths])
                    AverageReturn = np.mean(undiscounted_returns)
                    StdReturn = np.std(undiscounted_returns)
                    MaxReturn = np.max(undiscounted_returns)
                    MinReturn = np.min(undiscounted_returns)
                    self.rewards['average_discounted_return'].append(average_discounted_return)
                    self.rewards['AverageReturn'].append(AverageReturn)
                    self.rewards['StdReturn'].append(StdReturn)
                    self.rewards['MaxReturn'].append(MaxReturn)
                    self.rewards['MinReturn'].append(MinReturn)
                    print("AverageReturn: ",AverageReturn)
                    print("MaxReturn: ",MaxReturn)
                    print("MinReturn: ",MinReturn)
                    # print("returns: ",samples_data["returns"])
                    # print("valids: ",samples_data["valids"])

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
        if self.record_rewards:
            return dict(
                itr=itr,
                policy=self.policy,
                baseline=self.baseline,
                env=self.env,
                rewards=self.rewards,
            )
        else:
            return dict(
                itr=itr,
                policy=self.policy,
                baseline=self.baseline,
                env=self.env,
            )
