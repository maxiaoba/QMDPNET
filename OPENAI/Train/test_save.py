#!/usr/bin/env python3

from baselines import logger
# from baselines.common.cmd_util import make_atari_env,atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from Alg.a2c import learn
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
from Policy.qmdp_policy import QmdpPolicy
from Env.Atari.atari_wrapper import make_atari_env
from baselines.common.cmd_util import arg_parser
from Env import Atari

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env):
    if policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'qmdp':
        policy_fn = QmdpPolicy
    # env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
    env = make_atari_env(env_id, num_env, seed)
    # env = AtariEnv(mask_num=20 ,game='carnival')
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule,
        nsteps=200,
        save_path="./Data/a2cTest/")#,load_path="./Data/a2cTest/a2c_1.pkl")
    env.close()

def main():
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e4))
    parser.add_argument('--policy', help='Policy architecture', choices=['lstm','qmdp'], default='qmdp')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    args = parser.parse_args()
    logger.configure(dir="./Data/a2cTest/")
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_env=16)

if __name__ == '__main__':
    main()
