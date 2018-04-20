import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype

class QmdpPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, qmdp_param, reuse=False):
        nenv = nbatch // nsteps

        input_len = ob_space.shape
        input_shape = (nbatch, input_len)
        num_action = qmdp_param["num_action"]
        obs_len = qmdp_param["obs_len"]
        num_state = qmdp_param['num_state']
        num_obs = qmdp_param['num_obs']

        X = tf.placeholder(tf.uint8, input_shape) #obs+prev action
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, num_state]) #beliefs

        with tf.variable_scope("model", reuse=reuse):
            xs = batch_to_seq(X, nenv, nsteps)
            #dived xs to obs and pre_action
            obs = xs[:,:,0:obs_len]
            acts = xs[:,:,obs_len:]

            ms = batch_to_seq(M, nenv, nsteps)

            #build variabels
            planner_net = PlannerNet("planner",qmdp_param)
            filter_net = FilterNet("filter",qmdp_param)

            #calculate action value q, and belief bnew
            snew = self.filter_net.beliefupdate(obs, acts, S, nbatch)
            Q,_,_ = self.planner_net.VI(nbatch)

            # h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            # h5 = seq_to_batch(h5)

            pi = fc(h5, 'pi', nact) #actor
            vf = fc(h5, 'v', 1) #critic value function

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value