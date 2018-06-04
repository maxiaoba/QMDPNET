import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from Policy.lstm2_net import lstm

class Lstm2Policy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=16, reuse=False):
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
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #beliefs

        with tf.variable_scope("model", reuse=reuse):
            xs = batch_to_seq(X, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h = S[:,0:nlstm]
            c = S[:,nlstm:]

            self.lstm = lstm('lstm', input_len[0], nlstm)
            h5, snew = self.lstm.update(xs, ms, h, c)

            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

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