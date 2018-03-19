from tensorpack import graph_builder
import tensorflow as tf
import numpy as np

class PlannerNet(object):
    def __init__(self, qmdp_param, parent_layer=None):
        # self.params = params
        self.K = qmdp_param['K']
        self.num_action = qmdp_param['num_action']
        self.num_state = qmdp_param['num_state']
        self.f_R = F_R(self.num_state,self.num_action, parent_layer=parent_layer)
        self.f_pi = F_pi(self.num_action, self.num_action, parent_layer=parent_layer)
        self.f_T = F_T(self.num_state,self.num_action, name="planner_net", parent_layer=parent_layer)

    def VI(self, R0, V0):
        """
        builds neural network implementing value iteration. this is the first part of planner module. Fixed through time.
        inputs: map (batch x N x N) and goal(batch)
        returns: Q_K, and optionally: R, list of Q_i
        """
        # build reward model R
        # R = PlannerNet.f_R(map, goal, params.num_action,parent_layer=parent_layer)
        R = self.f_R.step(R0)

        # get transition model Tprime. It represents the transition model in the filter, but the weights are not shared.
        # kernel = FilterNet.f_T(params.num_action,parent_layer=parent_layer)

        # initialize value image
        # V = tf.zeros(map.get_shape().as_list() + [1])
        # V = tf.zeros((tf.shape(R0)[0],self.num_state))
        V = V0
        Q = None

        # repeat value iteration K times
        for i in range(self.K):
            # apply transition and sum
            # Q = tf.nn.conv2d(V, kernel, [1, 1, 1, 1], padding='SAME')
            Q = self.f_T.step(V)
            Q = Q + R
            V = tf.reduce_max(Q, axis=[2], keep_dims=False)

        return Q, V, R

    def policy(self, Q, b):
        """
        second part of planner module
        :param Q: input Q_K after value iteration
        :param b: belief at current step
        :param params: params
        :return: a_pred,  vector with num_action elements, each has the
        """
        # weight Q by the belief
        b_tiled = tf.tile(tf.expand_dims(b, 2), [1, 1, self.num_action])
        q = tf.multiply(Q, b_tiled)
        # sum over states
        q = tf.reduce_sum(q, 1, keep_dims=False)
        #self.printQ = tf.Print(q,[q],'q: ')
        self.q = q
        # low-level policy, f_pi
        # action_pred = PlannerNet.f_pi(q, params.num_action, parent_layer=parent_layer)
        action_pred = self.f_pi.step(q)
        self.action_pred = action_pred
        return action_pred


class FilterNet(object):
    def __init__(self, qmdp_param, parent_layer=None):
        # self.params = params
        self.num_action = qmdp_param['num_action']
        self.num_state = qmdp_param['num_state']
        self.f_T = F_T(self.num_state,self.num_action, name="filter_net", parent_layer=parent_layer)
        self.f_A = F_A()
        self.f_O = F_O(qmdp_param['obs_len'], parent_layer=parent_layer)
        self.f_Z = F_Z(qmdp_param['info_len'], self.num_state, parent_layer=parent_layer)

    def beliefupdate(self, Z, b, action, local_obs):
        """
        Belief update in the filter module with pre-computed Z.
        :param b: belief (b_i), [batch, N, M, 1]
        :param action: action input (a_i)
        :param obs: observation input (o_i)
        :return: updated belief b_(i+1)
        """
        # step 1: update belief with transition
        # get transition kernel (T)
        # kernel = FilterNet.f_T(params.num_action, parent_layer=parent_layer)

        # apply convolution which corresponds to the transition function in an MDP (f_T)
        # b = tf.expand_dims(b, -1)
        # b_prime = tf.nn.conv2d(b, kernel, [1, 1, 1, 1], padding='SAME')
        b_prime = self.f_T.step(b)

        # index into the appropriate channel of b_prime
        # w_A = FilterNet.f_A(action, params.num_action, parent_layer=parent_layer)
        w_A = self.f_A.step(action, self.num_action)
        w_A = w_A[:, None] #w_A to shape [batch,1,|A|]
        w_A = tf.to_float(w_A)
        b_prime_a = tf.reduce_sum(tf.multiply(b_prime, w_A), [2], keep_dims=False) # soft indexing

        #b_prime_a = tf.abs(b_prime_a) # TODO there was this line. does it make a difference with softmax?

        # step 2: update belief with observation
        # get observation probabilities for the obseravtion input by soft indexing
        # w_O = FilterNet.f_O(local_obs, parent_layer=parent_layer)
        w_O = self.f_O.step(local_obs)
        w_O = w_O[:,None] #tf.expand_dims(tf.expand_dims(w_O, axis=1), axis=1)
        Z_o = tf.reduce_sum(tf.multiply(Z, w_O), [2], keep_dims=False) # soft indexing

        b_next = tf.multiply(b_prime_a, Z_o)

        # step 3: normalize over the state space
        # add small number to avoid division by zero
        b_next = tf.div(b_next + 1e-8, tf.reduce_sum(b_next + 1e-8, [1], keep_dims=True))

        return b_next

class F_R(object):
    def __init__(self, num_state, num_action, parent_layer=None):
        # self.convlayers = ConvLayers(2, np.array([[3, 150, 'relu'], [1, num_action, 'lin']]), "R_conv", parent_layer=parent_layer)
        self.num_state = num_state
        self.num_action = num_action
        self.fclayers = FcLayers(num_state*num_action,np.array([[3*num_state*num_action, 'relu'], [num_state*num_action, 'lin']]), "R_fc",parent_layer=parent_layer)
    def step(self, R0):
        # theta = tf.stack([map, goal], axis=3)
        R = self.fclayers.step(R0)
        R = tf.reshape(R, [-1,self.num_state,self.num_action])
        return R

class F_T(object):
    def __init__(self, num_state, num_action, name, parent_layer=None):
        self.num_state = num_state
        self.num_action = num_action
        self.fclayers = FcLayers(num_state,np.array([[num_state*num_action, 'lin']]), name+"T_fc",parent_layer=parent_layer)

    def step(self, input):
        out = self.fclayers.step(input)
        out = tf.reshape(out, [-1,self.num_state,self.num_action])
        out = tf.nn.softmax(out,dim=1)
        return out

class F_Z(object):
    def __init__(self, info_len, num_state, parent_layer=None):
        # self.convlayers = ConvLayers(1, np.array([[3, 150, 'lin'], [1, 17, 'sig']]), "Z_conv", parent_layer=parent_layer)
        self.num_obs = 17
        self.num_state = num_state
        self.fclayers = FcLayers(info_len,np.array([[3*num_state*self.num_obs, 'relu'], [num_state*self.num_obs, 'sig']]), "Z_fc",parent_layer=parent_layer)
    def step(self, info):
        Z = self.fclayers.step(info)
        Z = tf.reshape(Z, [-1,self.num_state,self.num_obs])
        #Z = tf.nn.softmax(Z,dim=2)
        Z = tf.nn.softmax(Z) #default is -1=2, if use dim = 2, cause issues at some tf version
        return Z

class F_A(object):
    def __init__(self):
        pass
    def step(self, action, num_action):
        # identity function
        # w_A = tf.one_hot(action, num_action)
        # return w_A
        return action

class F_O(object):
    def __init__(self, obs_len, parent_layer=None):
        self.num_obs = 17
        self.fclayers = FcLayers(obs_len, np.array([[self.num_obs, 'tanh'], [self.num_obs, 'smax']]), names="O_fc", parent_layer=parent_layer)
    def step(self, local_obs):
        w_O = self.fclayers.step(local_obs)
        return w_O

class F_pi(object):
    def __init__(self, num_action_in, num_action_out, parent_layer=None):
        # self.fclayers = FcLayers(num_action_in, np.array([[num_action_out, 'smax']]), names="pi_fc", parent_layer=parent_layer)
        # Xiaobai: change nonlinear to smax
        pass
    def step(self, q):
        # return self.fclayers.step(q)
        return tf.nn.softmax(q, dim=-1)


# Helper function to construct layers conveniently

class ConvLayer(object):
    def __init__(self, input_size, kernel_size, filters, name, w_mean=0.0, w_std=None, addbias=True, strides=(1, 1, 1, 1), padding='SAME', parent_layer=None):
        self.input_size = input_size
        self.output_size = filters
        self.name = name
        self.addbias = addbias
        self.strides = strides
        self.padding = padding

        dtype = tf.float32
        if w_std is None:
            w_std = 1.0 / np.sqrt(float(input_size * kernel_size * kernel_size))

        initializer = tf.truncated_normal_initializer(mean=w_mean, stddev=w_std, dtype=dtype)    
        self.kernel = parent_layer.add_param_plain(initializer, [kernel_size, kernel_size, input_size, filters], name='w_'+name, trainable=True, regularizable=True)
        self.biases = None
        if addbias:
            initializer = tf.constant_initializer(0.0)
            biases = parent_layer.add_param_plain(initializer, [filters], name='b_' + name, trainable=True, regularizable=False)
            self.biases = biases

    def step(self, input):
        output = tf.nn.conv2d(input, self.kernel, strides=self.strides, padding=self.padding)

        if self.addbias:
            output = tf.nn.bias_add(output, self.biases)
        return output

class FcLayer(object):
    def __init__(self, input_size, output_size, name, w_mean=0.0, w_std=None, parent_layer=None):
        dtype = tf.float32
        self.input_size = input_size
        self.output_size = output_size
        self.name = name

        if w_std is None:
            # w_std = 1.0 / np.sqrt(float(np.prod(input.get_shape().as_list()[1])))
            w_std = 1.0 / np.sqrt(float(input_size))

        initializer = tf.truncated_normal_initializer(mean=w_mean, stddev=w_std, dtype=dtype)
        self.w = parent_layer.add_param_plain(initializer, [input_size, output_size], name='w_' + name, trainable=True, regularizable=True)

        # b = tf.get_variable("b_" + name, [output_size], initializer=tf.constant_initializer(0.0))

        initializer = tf.constant_initializer(0.0)
        self.b = parent_layer.add_param_plain(initializer, [output_size], name='b_' + name, trainable=True, regularizable=False)

    def step(self, input):
        output = tf.matmul(input, self.w) + self.b
        return output

class ConvLayers(object):
    def __init__(self, input_size, conv_params, names, parent_layer=None, **kwargs):
        self.input_size = input_size
        input_size = input_size
        output_size = 0
        self.convlayers = []
        self.activations = []
        for layer_i in range(conv_params.shape[0]):
            kernelsize = int(conv_params[layer_i][0])
            output_size = int(conv_params[layer_i][1])
            if isinstance(names, list):
                name = names[layer_i]
            else:
                name = names+'_%d'%layer_i
            convlayer = ConvLayer(input_size, kernelsize, output_size, name, parent_layer=parent_layer, **kwargs)
            self.convlayers.append(convlayer)
            self.activations.append(conv_params[layer_i][2])
            input_size = output_size
        self.output_size = output_size

    def step(self, input):
        output = input
        for convlayer, activation_name in zip(self.convlayers,self.activations):
            output = convlayer.step(output)
            output = activation(output, activation_name)
        return output

class FcLayers(object):
    def __init__(self, input_size, conv_params, names, parent_layer=None, **kwargs):
        self.input_size = input_size
        input_size = input_size
        output_size = 0
        self.fclayers = []
        self.activations = []
        for layer_i in range(conv_params.shape[0]):
            output_size = int(conv_params[layer_i][0])
            if isinstance(names, list):
                name = names[layer_i]
            else:
                name = names+'_%d'%layer_i
            fclayer = FcLayer(input_size, output_size, name, parent_layer=parent_layer, **kwargs)
            self.fclayers.append(fclayer)
            self.activations.append(conv_params[layer_i][-1])
            input_size = output_size
        self.output_size = output_size

    def step(self, input):
        output = input
        for fclayer, activation_name in zip(self.fclayers,self.activations):
            output = fclayer.step(output)
            output = activation(output, activation_name)
        return output

def activation(tensor, activation_name):
    """
    Apply activation function to tensor
    :param tensor: input tensor
    :param activation_name: string that defines activation [lin, relu, tanh, sig]
    :return: output tensor
    """
    if activation_name in ['l', 'lin']:
        pass
    elif activation_name in ['r', 'relu']:
        tensor = tf.nn.relu(tensor)
    elif activation_name in ['t', 'tanh']:
        tensor = tf.nn.tanh(tensor)
    elif activation_name in ['s', 'sig']:
        tensor = tf.nn.sigmoid(tensor)
    elif activation_name in ['sm', 'smax']:
        tensor = tf.nn.softmax(tensor, dim=-1)
    else:
        raise NotImplementedError

    return tensor

class XavierUniformInitializer(object):
    def __call__(self, shape, dtype=tf.float32, *args, **kwargs):
        if len(shape) == 2:
            n_inputs, n_outputs = shape
        else:
            receptive_field_size = np.prod(shape[:2])
            n_inputs = shape[-2] * receptive_field_size
            n_outputs = shape[-1] * receptive_field_size
        init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range, dtype=dtype)(shape)


class HeUniformInitializer(object):
    def __call__(self, shape, dtype=tf.float32, *args, **kwargs):
        if len(shape) == 2:
            n_inputs, _ = shape
        else:
            receptive_field_size = np.prod(shape[:2])
            n_inputs = shape[-2] * receptive_field_size
        init_range = math.sqrt(1.0 / n_inputs)
        return tf.random_uniform_initializer(-init_range, init_range, dtype=dtype)(shape)

