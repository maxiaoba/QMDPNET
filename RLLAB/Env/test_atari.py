from atari import AtariEnv
from sandbox.rocky.tf.envs.base import TfEnv

import sys
game_name = sys.argv[1]
mask_num = int(sys.argv[2])

env = TfEnv(AtariEnv(mask_num,game_name))

env.reset()
env.render()

d = False

timestep = 0.05

while not d:
	a = env.spec.action_space.sample()
	next_o, r, d, env_info = env.step(a)
	print(r,d)
	env.render()