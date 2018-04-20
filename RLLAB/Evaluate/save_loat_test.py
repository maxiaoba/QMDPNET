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
import numpy as np
#stub(globals())


env = TfEnv(GridBase())
env._wrapped_env.generate_grid=True
env._wrapped_env.generate_b0_start_goal=True
env.reset()
env._wrapped_env.generate_grid=False
env._wrapped_env.generate_b0_start_goal=False

env_img = env._wrapped_env.env_img
goal_img = env._wrapped_env.goal_img
b0_img = env._wrapped_env.b0_img
start_state = env._wrapped_env.start_state

params = dict(
    env=env,
)
# file = open('env.pkl', 'wb')
# dill.dump(params, file)
# with open('env.pkl', 'rb') as file:
#     params = dill.load(file)

joblib.dump(params,'env.pkl')
params=joblib.load('env.pkl')

env = params['env']
env.reset()
env_img2 = env._wrapped_env.env_img
goal_img2 = env._wrapped_env.goal_img
b0_img2 = env._wrapped_env.b0_img
start_state2 = env._wrapped_env.start_state
print(np.array_equal(env_img,env_img2))
print(np.array_equal(goal_img,goal_img2))
print(np.array_equal(b0_img,b0_img2))

