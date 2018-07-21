#!/usr/bin/env python3

from baselines import logger
# from baselines.common.cmd_util import make_atari_env,atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from Alg.a2c import learn_a2c
from Alg.ppo2 import learn_ppo2
# from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
from Policy.qmdp_policy import QmdpPolicy
from Policy.qmdp_policy_relu import QmdpPolicyRelu
from Policy.qmdp_policy_split import QmdpPolicySplit
from Policy.qmdp_policy_shallow import QmdpPolicyShallow
from Policy.qmdp_policy_k1 import QmdpPolicyK1
from Policy.qmdp_policy_dc import QmdpPolicyDc
from Policy.lstm_policy import LstmPolicy
from Policy.lstm2_policy import Lstm2Policy
from Policy.qmdp_policy_separate_value_network import QmdpSVNPolicy
from Env.Atari.atari_wrapper import make_atari_env
from baselines.common.cmd_util import arg_parser
from Env import Atari

def train(env_id, N_itr, seed, policy, lr, lrschedule, num_env, log_path, save_interval, alg):
    if policy == 'lstm16':
        policy_fn = LstmPolicy
    elif policy == 'lstm2':
        policy_fn = Lstm2Policy
    elif policy == 'qmdp':
        policy_fn = QmdpPolicy
    elif policy == 'qmdp_relu':
        policy_fn = QmdpPolicyRelu
    elif policy == 'qmdp_split':
        policy_fn = QmdpPolicySplit
    elif policy == 'qmdp_k1':
        policy_fn = QmdpPolicyK1
    elif policy == 'qmdp_shallow':
        policy_fn = QmdpPolicyShallow
    elif policy == 'qmdp_dc':
        policy_fn = QmdpPolicyDc
    elif policy == 'qmdp_svn':
        policy_fn = QmdpSVNPolicy
    # env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
    env = make_atari_env(env_id, num_env, seed)
    # env = AtariEnv(mask_num=20 ,game='carnival')
    if alg == 'a2c':
        learn_a2c(policy=policy_fn, env=env, seed=seed, N_itr=int(N_itr), lr=lr, lrschedule=lrschedule,
            nsteps=128,
            save_interval=save_interval,
            save_path=log_path)#,load_path="./Data/a2cTest/a2c_1.pkl")
    elif alg == 'ppo2':
        learn_ppo2(policy=policy_fn, env=env, seed=seed,nsteps=128,nminibatches=4,
            lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
            ent_coef=.01,
            lr=lambda f : f * lr,
            cliprange=lambda f : f * 0.1,
            N_itr=int(N_itr),
            save_interval=save_interval,
            save_path=log_path) #there are defalut values from original openai/baselines github
    env.close()

def main():
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='carnivalRam20-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--N_itr', type=int, default=int(2e4))
    parser.add_argument('--policy', help='Policy architecture', choices=['lstm16','lstm2',\
                        'qmdp','qmdp_relu','qmdp_split','qmdp_k1','qmdp_shallow','qmdp_dc','qmdp_svn'], default='qmdp')
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--save_interval', help='model save frequency', type=int, default=1000)
    parser.add_argument('--alg',help='training algorithm',choices=['a2c','ppo2'],default='a2c')
    args = parser.parse_args()
    log_path = "./Data/"+args.alg+'_'+args.policy+"_"+args.env+"lr"+str(args.lr)+"seed"+str(args.seed)+"_ID_12345/"
    # log_path = "./Data/a2cTest/"
    logger.configure(dir=log_path)
    train(args.env, N_itr=args.N_itr, seed=args.seed,
        policy=args.policy, lr=args.lr, lrschedule=args.lrschedule, num_env=16, log_path=log_path, save_interval=args.save_interval,
        alg=args.alg)

if __name__ == '__main__':
    main()
