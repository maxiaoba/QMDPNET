import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from Policy.qmdp_net_pifc2 import PlannerNet, FilterNet

class QmdpPolicyPifc2Split(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        nenv = nbatch // nsteps

        qmdp_param = {}
        qmdp_param['K'] = 3
        qmdp_param['obs_len'] = ob_space.shape[0]-ac_space.n
        qmdp_param['num_action'] = ac_space.n
        qmdp_param['num_state'] = 32
        qmdp_param['num_obs'] = 17

        input_len = ob_space.shape
        input_shape = (nbatch,) + input_len
        num_action = qmdp_param["num_action"]
        obs_len = qmdp_param["obs_len"]
        num_state = qmdp_param['num_state']
        num_obs = qmdp_param['num_obs']

        self.pdtype = make_pdtype(ac_space)
        X = tf.placeholder(tf.float32, input_shape) #[nbatch,obs+prev action]
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, num_state]) #beliefs

        with tf.variable_scope("model", reuse=reuse):
            obs, acts = tf.split(X, [obs_len,num_action],1)
            obs = batch_to_seq(obs, nenv, nsteps)
            acts = batch_to_seq(acts, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            #same as xs
            #ms has shape [nsteps,nenv]

            #build variabels
            self.planner_net = PlannerNet("planner",qmdp_param)
            self.filter_net = FilterNet("filter",qmdp_param)

            #calculate action value q, and belief bnew
            s_hist, snew = self.filter_net.beliefupdate(obs, acts, ms, S)
            # s_hist, snew, w_O, Z_o, b_prime_a, b_f = self.filter_net.beliefupdate(obs, acts, ms, S)
            #s_hist: [nstep,nenv,num_state]
            Q, _, _ = self.planner_net.VI(nbatch)

            # h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            # h5 = seq_to_batch(h5)

            #calculate action and value
            s_hist = seq_to_batch(s_hist) #[nbatch,num_state]
            q = self.planner_net.policy(Q,s_hist)

            self.pd, self.pi = self.pdtype.pdfromlatent(q)
            vf = fc(q, 'v', 1) #critic value function

            #pi = fc(h5, 'pi', nact) #actor
            #vf = fc(h5, 'v', 1) #critic value function

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        # self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)
        self.initial_state = np.ones((nenv, num_state), dtype=np.float32)/num_state

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})
            # a,b,c,d,q_val = sess.run([a0, v0, snew, neglogp0, q], {X:ob, S:state, M:mask})
            # print("q: ",q_val)
            # print("q shape: ",q_val.shape)
            # return a,b,c,d

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value