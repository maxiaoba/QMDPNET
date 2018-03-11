import tensorflow as tf
import joblib
from matplotlib import pyplot
import numpy as np
import os

sess = tf.Session()
sess.__enter__()

log_dir = "../MultiEnv/Data/QMDP"

names = []
for policy in os.listdir(log_dir):
    if not (policy == ".DS_Store"):
        names.append(policy[:-4:])

for name in names:
    tf.reset_default_graph()
    sess.close()
    sess = tf.Session()
    sess.__enter__()
    data = joblib.load(log_dir+'/'+name+'.pkl')
    agent = data['policy']
    max_path_length = 400
    success = 0
    path_lengths = np.array([])

    env_path = "../MultiEnv/"
    for i in range(20):
        data = joblib.load(env_path+'/TestEnv'+'/env_'+str(i)+'.pkl')
        env = data['env']
        env._wrapped_env.generate_grid=False
        env._wrapped_env.generate_b0_start_goal=False
        o = env.reset()
        agent.reset(env._wrapped_env.env_img, env._wrapped_env.goal_img, env._wrapped_env.b0_img)
        # agent.reset()
        path_length = 0

        while True:
            a, agent_info = agent.get_action(o)
            next_o, r, d, env_info = env.step(a)
            path_length += 1
            if d:
             	break
            o = next_o

        if path_length < max_path_length:
            success += 1
            path_lengths = np.append(path_lengths,path_length)
    mean_path_length = np.mean(path_lengths)
    print(name)
    print('success: ',success)
    print('mean length: ',mean_path_length)

