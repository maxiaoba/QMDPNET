import os.path as osp
import time
import joblib
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.runners import AbstractEnvRunner
from baselines.common import tf_util

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import cat_entropy, mse

# combine all the losses from all the different moving pieces
class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.make_session()
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch]) # perhaps this is advantage?
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            ############################ place where advantage is calculated
            ####################################################################
            advs = rewards - values
            ####################################################################
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            # policy_loss, value_loss, policy_entropy, _ = sess.run(
            #     [pg_loss, vf_loss, entropy, _train],
            #     td_map
            # )
            policy_loss, value_loss, policy_entropy, grads_val, _ = sess.run(
                [pg_loss, vf_loss, entropy, [x[0] for x in grads],_train],
                td_map
            )

            return policy_loss, value_loss, policy_entropy, grads_val

        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load

        self.sess = sess
        self.params = params
        self.grads = grads

        tf.global_variables_initializer().run(session=sess)

class Runner(AbstractEnvRunner):

    def __init__(self, env, model, nsteps=5, gamma=0.99):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            ##################################################### this is where values come from
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            # print(rewards.shape): (env_num,)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        # print(mb_rewards.shape): [num_env,nsteps]
        avg_reward = mb_rewards.sum(axis=1).mean()
        max_reward = mb_rewards.sum(axis=1).max()
        min_reward = mb_rewards.sum(axis=1).min()

        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            # print(len(rewards)): nsteps, each entry is the accumulated discounted reward at time t
            mb_rewards[n] = rewards

        # print(mb_rewards.shape): (nenv,ntimesteps), each env's accumulated discounter reward at each timestep
        avg_dis_reward = mb_rewards.mean(axis=0)[0]
        
        mb_rewards = mb_rewards.flatten()
        # print(mb_rewards.shape): (nenv*ntimesteps,)
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()

        info = dict()
        info['avg_reward'] = avg_reward
        info['avg_dis_reward'] = avg_dis_reward
        info['max_reward'] = max_reward
        info['min_reward'] = min_reward
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, info

def learn_a2c(policy, env, seed, nsteps=5, N_itr=1e4, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, 
            log_interval=1,
            save_interval=10,save_path="./Data/a2c",load_path=None):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    total_timesteps = N_itr*nenvs*nsteps

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)

    if load_path is not None:
        model.load(load_path)
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

    nbatch = nenvs*nsteps
    tstart = time.time()

    # writer = tf.summary.FileWriter(logdir=save_path)

    for update in range(N_itr):
        obs, states, rewards, masks, actions, values, info = runner.run() # here, values are the discounted rewards, i believe

        # policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        policy_loss, value_loss, policy_entropy, grads_val = model.train(obs, states, rewards, masks, actions, values)

        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)


        if update % log_interval == 0 or update == 1:
            params = [x[1] for x in model.grads]
            params_val = model.sess.run(model.params)
            for param,grad_val,param_val in zip(params,grads_val,params_val):
                # print(param.name+" value: ",np.mean(np.abs(param_val)))
                # print(param.name+"gradient: ",np.max(np.abs(grad_val)))
                logger .record_tabular(param.name+" gradient: ",np.max(np.abs(grad_val)))


            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("explained_variance", float(ev))

            for key in info.keys():
                logger.record_tabular(key,info[key])
            logger.dump_tabular()
        if update % save_interval == 0 or update == 1:
            model.save(save_path+"a2c_"+str(update)+".pkl")
    # writer.add_graph(model.sess.graph)
    # writer.close()
    env.close()
