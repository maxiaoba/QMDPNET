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

log_dir = "./Data/MultiMaze/TestEnv"
envNum = 10
for i in range(envNum):
	params = joblib.load(log_dir+'/env_'+str(i)+'.pkl')
	env = params['env']
	env_params = env.get_param_values()
	params = joblib.load(log_dir+'/env_'+str(0)+'.pkl')
	env = params['env']
	env.set_param_values(env_params)

	env_img = env._wrapped_env.env_img
	goal_img = env._wrapped_env.goal_img
	b0_img = env._wrapped_env.b0_img
	start_state = env._wrapped_env.start_state
	state = env._wrapped_env.state

	show_img = np.copy(env_img)
	start_coord = env._wrapped_env.state_lin_to_bin(env._wrapped_env.start_state)
	show_img[start_coord[0]][start_coord[1]] = 2

	show_img = show_img + 3 * goal_img

	current_coord = env._wrapped_env.state_lin_to_bin(state)
	show_img[current_coord[0]][current_coord[1]] = 4
	# make a color map of fixed colors
	cmap = mpl.colors.ListedColormap(['white','black','red','blue','yellow'])
	bounds=[-0.5,0.5,1.5,2.5,3.5,4.5]
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
	fig = pyplot.figure(1)
	# tell imshow about color map so that only set colors are used
	img = pyplot.imshow(show_img,interpolation='nearest',
	                    cmap = cmap,norm=norm)

	# make a color bar
	# pyplot.colorbar(img,cmap=cmap,
	                # norm=norm,boundaries=bounds,ticks=[0,1,2,3,4])

	# pyplot.show()
	fig.savefig(log_dir+'/Map_'+str(i)+'.pdf')
	pyplot.close(fig)


