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
from matplotlib import pyplot
import matplotlib as mpl

#stub(globals())

# log_dir = "./Data/FixMapStartState"
# params = joblib.load(log_dir+'/env.pkl')
# env = params['env']

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
print(b0_img)
print(env_img)
print(env.step(4))
print(env.step(3))
print(env.step(2))
print(env.step(1))
print(env.step(0))
show_img = np.copy(env_img)
start_coord = env._wrapped_env.state_lin_to_bin(start_state)
show_img[start_coord[0]][start_coord[1]] = 2

show_img = show_img + 3 * goal_img

show_img = show_img + 4 * (b0_img>0) + b0_img
# make a color map of fixed colors
cmap = mpl.colors.ListedColormap(['white','black','red','blue','white','yellow','orange','purple','red'])
bounds=[0,1,2,3,4,4.001,4.3,4.7,5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# tell imshow about color map so that only set colors are used
img = pyplot.imshow(show_img,interpolation='nearest',
                    cmap = cmap,norm=norm)

# make a color bar
pyplot.colorbar(img,cmap=cmap,
                norm=norm,boundaries=bounds,ticks=[0,1,2,3,4,4.001,4.3,4.7,5])

pyplot.show()
