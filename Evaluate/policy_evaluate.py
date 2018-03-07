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
log_dir = "./Data/Test2"
data = joblib.load(log_dir+'/params.pkl')
# env = data['env']
agent = data['policy']
max_path_length = 400
animated = True
speedup = 1

# data = joblib.load('./env.pkl')
env = data['env']
# env = TfEnv(GridBase())
# env._wrapped_env.generate_grid=True
# env._wrapped_env.generate_b0_start_goal=True
# env.reset()
# env._wrapped_env.generate_grid=False
# env._wrapped_env.generate_b0_start_goal=False

assert(env._wrapped_env.generate_grid==False)
assert(env._wrapped_env.generate_b0_start_goal==False)
o = env.reset()
agent.reset(env._wrapped_env.env_img, env._wrapped_env.goal_img, env._wrapped_env.b0_img)
# agent.reset()
path_length = 0
pyplot.show()
if animated:
    env.render()

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