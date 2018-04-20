from Env.cartpole import CartPoleEnv
from sandbox.rocky.tf.envs.base import TfEnv

env = TfEnv(CartPoleEnv(2))

env.reset()
env.render()

d = False

timestep = 0.05

while not d:
	a = env.spec.action_space.sample()
	next_o, r, d, env_info = env.step(a)
	print(next_o)
	env.render()