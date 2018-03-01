from grid_env import GridBase
import numpy as np
from sandbox.rocky.tf.envs.base import TfEnv

# env = GridBase(generate_b0_start_goal = False, generate_grid = True)
# obs = env.reset()
# print(env.env_img.shape, env.goal_img.shape, env.b0_img.shape)


env = GridBase()
env.reset()
tfenv = TfEnv(env)
print(tfenv.get_param_values())