import time
from rllab.algos.base import RLAlgorithm
from sandbox.rocky.tf.policies.base import Policy
from Algo.qmdp_sampler import QMDPSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
import numpy as np

from rllab.misc import logger
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.misc import tensor_utils
from rllab.core.serializable import Serializable
import tensorflow as tf
import joblib


class VPG_t(BatchPolopt, Serializable):
    """
    Vanilla Policy Gradient.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            env_path,
            env_num,
            env_keep_itr,
            optimizer=None,
            optimizer_args=None,
            transfer=True,
            record_rewards=True,
            rewards=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_epochs=1,
            )
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
            optimizer = FirstOrderOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.opt_info = None

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
        self.env_path = env_path
        self.env_num = env_num
        self.env_keep_itr = env_keep_itr
        super(VPG_t, self).__init__(env=env, policy=policy, baseline=baseline, sampler_cls=QMDPSampler,sampler_args=dict(), **kwargs)

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
            name='advantage',
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
        # state_info_vars include prev_action is recurrent, same for start_info_vars_list
        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        logli = dist.log_likelihood_sym(action_var, dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        if is_recurrent:
            surr_obj = - tf.reduce_sum(logli * advantage_var * valid_var) / tf.reduce_sum(valid_var)
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            max_kl = tf.reduce_max(kl * valid_var)
        else:
            surr_obj = - tf.reduce_mean(logli * advantage_var)
            mean_kl = tf.reduce_mean(kl)
            max_kl = tf.reduce_max(kl)

        input_list = [obs_var, action_var, advantage_var] + state_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(loss=surr_obj, target=self.policy, inputs=input_list)

        f_kl = tensor_utils.compile_function(
            inputs=input_list + old_dist_info_vars_list,
            outputs=[mean_kl, max_kl],
        )
        self.opt_info = dict(
            f_kl=f_kl,
        )

    @overrides
    def optimize_policy(self, itr, samples_data):
        logger.log("optimizing policy")
        inputs = ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        )
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        # state_info_keys is prev_action
        # so agent_infos should include prev_action
        # agent_infos from policy.get_action
        inputs += tuple(state_info_list)
        if self.policy.recurrent:
            inputs += (samples_data["valids"],)
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        loss_before = self.optimizer.loss(inputs)
        self.optimizer.optimize(inputs)
        loss_after = self.optimizer.loss(inputs)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)

        mean_kl, max_kl = self.opt_info['f_kl'](*(list(inputs) + dist_info_list))
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('MaxKL', max_kl)

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
                if itr % self.env_keep_itr == 0:
                    env_num = np.random.randint(self.env_num)
                    params = joblib.load(self.env_path+'/env_'+str(env_num)+'.pkl')
                    self.env = params['env']
                logger.log("Obtaining samples at env %d" % env_num)
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
