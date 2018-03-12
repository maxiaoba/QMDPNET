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
import time
import dill
from matplotlib import pyplot

sess = tf.Session()
sess.__enter__()
log_dir = "../MultiEnv/Data/QMDP/obs_1goal20step0stay_1_kdist_01_keep1.pkl"
data = joblib.load(log_dir)
# env = data['env']
agent = data['policy']
max_path_length = 400
animated = True
speedup = 1

pyplot.show()
for i in range(20):
    print('load env: ',i)
    data = joblib.load('../MultiEnv/TestEnv'+'/env_'+str(i)+'.pkl')
    env = data['env']
    env._wrapped_env.generate_grid=False
    env._wrapped_env.generate_b0_start_goal=False
    print('reset env')
    o = env.reset()
    print('reset agent')
    agent.reset(env._wrapped_env.env_img, env._wrapped_env.goal_img, env._wrapped_env.b0_img)
    # agent.reset()
    path_length = 0
    print('first render')
    if animated:
        env.render()
    if i == 0:
        pyplot.pause(20)

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        print('step: '+str(path_length)+'action: '+str(a)+' d: '+str(d))
        path_length += 1
        if d:
        	# o = env.reset()
        	# agent.reset()
         	# path_length = 0
         	break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            pyplot.pause(timestep / speedup)