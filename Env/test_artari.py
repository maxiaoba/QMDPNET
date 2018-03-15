from atari import AtariEnv
from sandbox.rocky.tf.envs.base import TfEnv

env = TfEnv(AtariEnv())

env.reset()
env.render()

d = False

timestep = 0.05

while not d:
	a = env.spec.action_space.sample()
	next_o, r, d, env_info = env.step(a)
	print(next_o)
	env.render()