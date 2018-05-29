from gym.envs.registration import register


games = ['carnival','star_gunner']
for game in games:
	register(
	    # id='fooNoFrameskip-v0',
	    id=game+'Ram20-v0',
	    entry_point='Env.Atari.atari:AtariEnv',
	    kwargs={'game': game, 'obs_type': 'ram', 'mask_num':20,'frameskip': 1},
	    max_episode_steps=10000,
	    nondeterministic=False,
	)