import tensorflow as tf
import joblib
import time
import numpy as np
from matplotlib import pyplot
import matplotlib as mpl

sess = tf.Session()
sess.__enter__()
log_dir = "../MultiEnv/Data/Itr1e4/Baseline/b_obs_1goal20step0stay_1_kdist_01_keep1.pkl"
data = joblib.load(log_dir)
# env = data['env']
agent = data['policy']
max_path_length = 400

path = "../MultiEnv/Data/Itr1e4/Baseline/keep1_path/"

env = data['env']
env._wrapped_env.generate_grid=True
env._wrapped_env.generate_b0_start_goal=True
env.reset()
env._wrapped_env.generate_grid=False
env._wrapped_env.generate_b0_start_goal=False

for i in range(10):
    data = joblib.load('../MultiEnv/TestEnv'+'/env_'+str(i)+'.pkl')
    env_ref = data['env']
    grid = env_ref._wrapped_env.grid
    b0 = env_ref._wrapped_env.b0
    start_state = env_ref._wrapped_env.start_state
    goal_state = env_ref._wrapped_env.goal_state
    env._wrapped_env.__init__(env._wrapped_env.params,grid=grid,b0=b0,start_state=start_state,goal_state=goal_state)
    env._wrapped_env.generate_grid=False
    env._wrapped_env.generate_b0_start_goal=False

    o = env.reset()
    agent.reset()
    path_length = 0

    env_img = env._wrapped_env.env_img
    goal_img = env._wrapped_env.goal_img
    b0_img = env._wrapped_env.b0_img
    start_state = env._wrapped_env.start_state

    show_img = np.copy(env_img)
    start_coord = env._wrapped_env.state_lin_to_bin(env._wrapped_env.start_state)
    show_img[start_coord[0]][start_coord[1]] = 2

    show_img = show_img + 3 * goal_img

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        path_length += 1
        if d:
         	break
        o = next_o

        state = env._wrapped_env.state
        current_coord = env._wrapped_env.state_lin_to_bin(state)
        show_img[current_coord[0]][current_coord[1]] = 4
        if goal_img[current_coord[0]][current_coord[1]] == 1:
            show_img[current_coord[0]][current_coord[1]] = 3

    show_img[start_coord[0]][start_coord[1]] = 2
    # make a color map of fixed colors

    cmap = mpl.colors.ListedColormap(['white','black','blue','red','yellow'])
    bounds=[-0.5,0.5,1.5,2.5,3.5,4.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig = pyplot.figure(1)
    # tell imshow about color map so that only set colors are used
    img = pyplot.imshow(show_img,interpolation='nearest',
                        cmap = cmap,norm=norm)
    
    fig.savefig(path+'Baseline_'+str(i)+'.png')
    pyplot.close(fig)


