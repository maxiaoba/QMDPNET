import numpy as np
from matplotlib import pyplot
import matplotlib as mpl

def plot_env(env,save=False,show=False,path=None):
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

	
	if save:
		fig.savefig(path)
	if show:
		pyplot.show()
	pyplot.close(fig)