import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from Policy.qmdp_net import PlannerNet, FilterNet

class QmdpSVNPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=16, reuse=False):
        nenv = nbatch // nsteps

        qmdp_param = {}
        qmdp_param['K'] = 3
        qmdp_param['obs_len'] = ob_space.shape[0]-ac_space.n
        qmdp_param['num_action'] = ac_space.n
        qmdp_param['num_state'] = 32
        qmdp_param['num_obs'] = 17

        input_len = ob_space.shape
        input_shape = (nbatch,) + input_len # [nbatch, input_length]
        num_action = qmdp_param["num_action"]
        obs_len = qmdp_param["obs_len"]
        num_state = qmdp_param['num_state']
        num_obs = qmdp_param['num_obs']

        self.pdtype = make_pdtype(ac_space)
        X = tf.placeholder(tf.float32, input_shape) #[nbatch,obs+prev action]
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, num_state+2*nlstm]) # belief state (for each env)
        # S is belief state concatenated with initial hidden and cell states for vf lstm

        with tf.variable_scope("model", reuse=reuse):
            xs = batch_to_seq(X, nenv, nsteps)
            #xs originaly [nbatch,input_len]
            #reshape xs to [nenv,nsteps,input_len]
            #split xs along axis=1 to nsteps
            #xs becomes [nsteps,nenv,input_len] 
            #divide xs to obs and pre_action
            obs = [x[:,0:obs_len] for x in xs]
            acts = [x[:,obs_len:] for x in xs]
            ms = batch_to_seq(M, nenv, nsteps)
            #same as xs
            #ms has shape [nsteps,nenv]
            bi = S[:,0:num_state] # initial/previous belief
            hi = S[:,num_state:] # initial/previous hidden unit

            #build variabels
            self.planner_net = PlannerNet("planner",qmdp_param)
            self.filter_net = FilterNet("filter",qmdp_param)

            #calculate action value q, and belief bnew
            # s_hist is really belief state history, so really belief history
            # snew is the newest belief
            s_hist, snew = self.filter_net.beliefupdate(obs, acts, ms, bi)
            # s_hist, snew, w_O, Z_o, b_prime_a, b_f = self.filter_net.beliefupdate(obs, acts, ms, S)
            #s_hist: [nstep,nenv,num_state]
            # snew: [nenv, num_state]
            Q, _, _ = self.planner_net.VI(nbatch)
            # Q: [nbatches, num_state, num_action]

            # h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            # h5 = seq_to_batch(h5)

            #calculate action and value
            s_hist = seq_to_batch(s_hist) #[nbatch,num_state] (belief history)
            q = self.planner_net.policy(Q,s_hist) # [num_batch, num_action]

            # separate value function for baseline
            # takes in sequence of observations and actions and returns values of the belief states
            # in the belief history
            vn_scope = "value_network"
            # hi is of dim 2*nlstm
            # xs is the obs and acts concatenated
            # TODO: What shape do I want xs to be in? [nsteps, nenv, nobs+nacts], which is what it is! 
            # TODO: And what shape do I want chnew to be? [nenv, nlstm]
            h_hist, chnew = lstm(xs, ms, hi, vn_scope, nlstm)
            h_hist = tf.convert_to_tensor(h_hist, dtype=tf.float32)
            # h_hist.shape: (nstep, nenv, nlstm)
            # chnew.shape: (nenv, 2*nlstm)
            Snew = tf.concat(axis=1, values=[snew, chnew])
            # stack snew and chnew
            ############### baseline value function #####################################
            #############################################################################
            self.pd, self.pi = self.pdtype.pdfromlatent(q)
            # input dim of fc: shape(q)[1] = num_action, output dim of fc: 1
            #vf = fc(q, 'v', 1) #critic value function, output shape: [num_batch, 1]
            vf = fc(fc(fc(h_hist, 'v1', nlstm), 'v2', nlstm), 'v3', 1)
            #############################################################################

            #pi = fc(h5, 'pi', nact) #actor
            #vf = fc(h5, 'v', 1) #critic value function

        v0 = vf[:, 0] # reduce dims from [num_batch, 1] to [num_batch, ]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        # self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)
        self.initial_state = np.ones((nenv, num_state), dtype=np.float32)/num_state

        def step(ob, belief_state, mask):
            return sess.run([a0, v0, Snew, neglogp0], {X:ob, S:belief_state, M:mask})
            # a,b,c,d,q_val = sess.run([a0, v0, snew, neglogp0, q], {X:ob, S:state, M:mask})
            # print("q: ",q_val)
            # print("q shape: ",q_val.shape)
            # return a,b,c,d

        def value(ob, belief_state, mask):
            return sess.run(v0, {X:ob, S:belief_state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value