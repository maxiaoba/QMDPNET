import os
import gym
from gym.wrappers import FlattenDictWrapper
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import NoopResetEnv,MaxAndSkipEnv,EpisodicLifeEnv,FireResetEnv,ScaledFloatFrame,ClipRewardEnv,FrameStack
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)

from baselines.common.cmd_util import arg_parser

def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            #return wrap_deepmind(env, **wrapper_kwargs)
            return env
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def make_atari(env_id):
    env = gym.make(env_id)
    # assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    # env = MaxAndSkipEnv(env, skip=4)
    return env

def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    # env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env