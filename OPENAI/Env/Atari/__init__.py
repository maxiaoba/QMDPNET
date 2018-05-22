from gym.envs.registration import register

register(
    # id='fooNoFrameskip-v0',
    id='carnivalRam20-v0',
    entry_point='Env.Atari.atari:AtariEnv',
    kwargs={'game': 'carnival', 'obs_type': 'ram', 'mask_num':20,'frameskip': 1},
    max_episode_steps=10000,
    nondeterministic=False,
)