from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from Alg.a2c import learn_a2c
from Alg.ppo2 import learn_ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
from baselines.common.cmd_util import arg_parser

def train(env_id, N_itr, seed, policy, lr, lrschedule, num_env, log_path, save_interval, alg):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
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
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--N_itr', type=int, default=int(2e4))
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--save_interval', help='model save frequency', type=int, default=1000)
    parser.add_argument('--alg',help='training algorithm',choices=['a2c','ppo2'],default='a2c')
    args = parser.parse_args()
    log_path = "./Data/"+args.alg+'_'+args.policy+"_"+args.env+"lr"+str(args.lr)+"seed"+str(args.seed)
    # log_path = "./Data/a2cTest/"
    logger.configure(dir=log_path)
    train(args.env, N_itr=args.N_itr, seed=args.seed,
        policy=args.policy, lr=args.lr, lrschedule=args.lrschedule, num_env=16, log_path=log_path, save_interval=args.save_interval,
        alg=args.alg)

if __name__ == '__main__':
    main()

