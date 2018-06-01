import tensorflow as tf
import numpy as np

class PlannerNet(object):
    def __init__(self, name, qmdp_param):
        # self.params = params
        self.K = qmdp_param['K']
        self.num_action = qmdp_param['num_action']
        self.num_state = qmdp_param['num_state']
        self.f_R = F_R(self.num_state, self.num_action, name)
        self.f_pi = F_pi(self.num_action, self.num_action, name)
        self.f_T = F_T_planner(self.num_state, self.num_action, name)
        self.R0 = create_param(tf.constant_initializer(0.1), (self.num_state*self.num_action,), name="R0", trainable=True, regularizable=False)
        self.V0 = create_param(tf.constant_initializer(0.1), (self.num_state), name="V0", trainable=True, regularizable=False)

    def VI(self,n_batches):
        
        R0 = self.R0
        R0 = tf.tile(tf.reshape(R0, (1,self.num_state*self.num_action)),(n_batches,1))
        R0 = tf.to_float(R0)

        V0 = self.V0
        V0 = tf.tile(tf.reshape(V0, (1,self.num_state)),(n_batches,1))
        V0 = tf.to_float(V0)

        R = self.f_R.step(R0)

        V = V0
        Q = None

        # repeat value iteration K times
        for i in range(self.K):
            Q = self.f_T.step(V)
            Q = Q + R
            V = tf.reduce_max(Q, axis=[2], keep_dims=False)

        return Q, V, R

    def policy(self, Q, b):
        b = tf.to_float(b)

        b_tiled = tf.tile(tf.expand_dims(b, 2), [1, 1, self.num_action])
        q = tf.multiply(Q, b_tiled)
        # sum over states
        q = tf.reduce_sum(q, 1, keep_dims=False)
        #self.printQ = tf.Print(q,[q],'q: ')
        # self.q = q
        # low-level policy, f_pi
        # action_pred = PlannerNet.f_pi(q, params.num_action, parent_layer=parent_layer)
        # action_pred = self.f_pi.step(q)
        # self.action_pred = action_pred
        return q


class FilterNet(object):
    def __init__(self, name, qmdp_param):
        # self.params = params
        self.num_action = qmdp_param['num_action']
        self.num_state = qmdp_param['num_state']
        self.num_obs = qmdp_param['num_obs']
        self.f_T = F_T_filter(self.num_state,self.num_action, name)
        self.f_A = F_A(name)
        self.f_O = F_O(qmdp_param['obs_len'],qmdp_param['num_obs'], name)
        self.f_Z = F_Z(qmdp_param['num_obs'], self.num_state, name)
        self.z_os = create_param(tf.constant_initializer(1.0/self.num_obs), (self.num_state*self.num_obs), name="z_os", trainable=True, regularizable=False)
        # self.z_os = create_param(tf.truncated_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32), (self.num_state*self.num_obs), name="z_os", trainable=True, regularizable=False)

    def beliefupdate(self, local_obs, actions, ms, b):
        """
        Belief update in the filter module with pre-computed Z.
        """
        #local_obs: [nsteps,nenv,obs_len]
        #actions: [nsteps,nenv,obs_len]
        #ms: [nsteps,nenv] masks
        #b: [nenv, num_state]
        # nsteps, nenv, obs_len = local_obs.get_shape()
        nsteps = len(local_obs)
        nenv,obs_len = local_obs[0].get_shape()

        z_os= self.z_os
        z_os = tf.tile(tf.reshape(z_os, (1,self.num_state*self.num_obs)),(int(nenv),1))
        z_os = tf.to_float(z_os)
        Z = self.f_Z.step(z_os) #[nenv,num_state,num_obs]

        for idx, (local_ob,action,m) in enumerate(zip(local_obs,actions,ms)):
            #local_ob: [nenv,obs_len]
            #action: [nenv, action_num]
            #m: [nenv,1]

            b = b*(1-m)
            b_prime = self.f_T.step(b) #[nenv,num_state,num_action]
            # index into the appropriate channel of b_prime
            w_A = self.f_A.step(action, self.num_action) #[nenv, action_num]
            w_A = w_A[:, None] #w_A to shape [nenv,1,action_num]
            w_A = tf.to_float(w_A)
            b_prime_a = tf.reduce_sum(tf.multiply(b_prime, w_A), [2], keep_dims=False) # hard indexing [nenv,num_state]

            b_prime_a = tf.nn.relu(b_prime_a) #qmdp2

            # step 2: update belief with observation
            # get observation probabilities for the obseravtion input by soft indexing

            w_O = self.f_O.step(local_ob) #[nenv,num_obs]
            w_O = w_O[:,None] #[nenv,1,num_obs]
            Z_o = tf.reduce_sum(tf.multiply(Z, w_O), [2], keep_dims=False) # soft indexing [nenv,num_state]

            b = tf.multiply(b_prime_a, Z_o) #[nenv,num_state]

            b = tf.nn.relu(b) #qmdp2

            # step 3: normalize over the state space
            # add small number to avoid division by zero
            # b = tf.div(b + 1e-8, tf.reduce_sum(b + 1e-8, [1], keep_dims=True)) #[nenv,num_state] #qmdp2
            local_obs[idx] = b #do this just to reduce memory usage, now local_obs becomes belief history
         #local_obs now has dimension [nstep,nenv,num_state]
        return local_obs,b
        # return local_obs,b,w_O, Z_o, b_prime_a, b

class F_R(object):
    def __init__(self, num_state, num_action,name):
        self.num_state = num_state
        self.num_action = num_action
    def step(self, R0):
        R = tf.reshape(R0, [-1,self.num_state,self.num_action])
        return R

class F_T_filter(object):
    def __init__(self, num_state, num_action,name):
        self.num_state = num_state
        self.num_action = num_action
        w_std = 1.0 / np.sqrt(float(num_state))
        w_mean = 0.0
        dtype = tf.float32
        input_size = num_state
        output_size = num_state*num_action
        initializer = tf.truncated_normal_initializer(mean=w_mean, stddev=w_std, dtype=dtype)
        self.w = create_param(initializer, [input_size, output_size], name=name+"-w_f_T", trainable=True, regularizable=False)
    def step(self, input):
        weight = tf.reshape(self.w, [self.num_state, self.num_state, self.num_action])
        # weight = tf.nn.softmax(weight, dim=1)
        weight = tf.nn.relu(weight) #qmdp2
        weight = tf.reshape(weight, [self.num_state, self.num_state*self.num_action])
        out = tf.matmul(input, weight)
        out = tf.reshape(out, [-1,self.num_state,self.num_action])
        return out

class F_T_planner(object):
    def __init__(self, num_state, num_action,name):
        self.num_state = num_state
        self.num_action = num_action
        w_std = 1.0 / np.sqrt(float(num_state))
        w_mean = 0.0
        dtype = tf.float32
        input_size = num_state
        output_size = num_state*num_action
        initializer = tf.truncated_normal_initializer(mean=w_mean, stddev=w_std, dtype=dtype)
        self.w = create_param(initializer, [input_size, output_size], name=name+"-w_f_T", trainable=True, regularizable=False)
    def step(self, input):
        weight = tf.reshape(self.w, [self.num_state, self.num_state, self.num_action])
        # weight = tf.nn.softmax(weight, dim=0)
        weight = tf.nn.relu(weight) #qmdp2
        weight = tf.reshape(weight, [self.num_state, self.num_state*self.num_action])
        out = tf.matmul(input, weight)
        out = tf.reshape(out, [-1,self.num_state,self.num_action])
        return out

class F_Z(object):
    def __init__(self, num_obs, num_state,name):
        self.num_obs = num_obs
        self.num_state = num_state
    def step(self, Z_os):
        Z = tf.reshape(Z_os, [-1,self.num_state,self.num_obs])
        # Z = tf.nn.softmax(Z) #default is -1=2, if use dim = 2, cause issues at some tf version
        Z = tf.nn.relu(Z) #qmdp2
        return Z

class F_A(object):
    def __init__(self,name):
        pass
    def step(self, action, num_action):
        # identity function
        # w_A = tf.one_hot(action, num_action)
        # return w_A
        return action

class F_O(object):
    def __init__(self, obs_len, num_obs,name):
        # self.fclayers = FcLayers(obs_len, np.array([[num_obs, 'tanh'], [num_obs, 'smax']]), names=name+"-O_fc")
        self.fclayers = FcLayers(obs_len, np.array([[num_obs, 'tanh'], [num_obs, 'relu']]), names=name+"-O_fc")
    def step(self, local_obs):
        w_O = self.fclayers.step(local_obs)
        return w_O

class F_pi(object):
    def __init__(self, num_action_in, num_action_out,name):
        # self.fclayers = FcLayers(num_action_in, np.array([[num_action_out, 'smax']]), names=name+"pi_fc")
        # Xiaobai: change nonlinear to smax
        pass
    def step(self, q):
        # return self.fclayers.step(q)
        return tf.nn.softmax(q, dim=-1)


# Helper function to construct layers conveniently

class ConvLayer(object):
    def __init__(self, input_size, kernel_size, filters, name, w_mean=0.0, w_std=None, addbias=True, strides=(1, 1, 1, 1), padding='SAME'):
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
        self.kernel = create_param(initializer, [kernel_size, kernel_size, input_size, filters], name='w_'+name, trainable=True, regularizable=True)
        self.biases = None
        if addbias:
            initializer = tf.constant_initializer(0.0)
            biases = create_param(initializer, [filters], name='b_' + name, trainable=True, regularizable=False)
            self.biases = biases

    def step(self, input):
        output = tf.nn.conv2d(input, self.kernel, strides=self.strides, padding=self.padding)

        if self.addbias:
            output = tf.nn.bias_add(output, self.biases)
        return output

class FcLayer(object):
    def __init__(self, input_size, output_size, name, w_mean=0.0, w_std=None):
        dtype = tf.float32
        self.input_size = input_size
        self.output_size = output_size
        self.name = name

        if w_std is None:
            # w_std = 1.0 / np.sqrt(float(np.prod(input.get_shape().as_list()[1])))
            w_std = 1.0 / np.sqrt(float(input_size))

        initializer = tf.truncated_normal_initializer(mean=w_mean, stddev=w_std, dtype=dtype)
        self.w = create_param(initializer, [input_size, output_size], name='w_' + name, trainable=True, regularizable=True)

        # b = tf.get_variable("b_" + name, [output_size], initializer=tf.constant_initializer(0.0))

        initializer = tf.constant_initializer(0.0)
        self.b = create_param(initializer, [output_size], name='b_' + name, trainable=True, regularizable=False)

    def step(self, input):
        output = tf.matmul(input, self.w) + self.b
        return output

class ConvLayers(object):
    def __init__(self, input_size, conv_params, names, **kwargs):
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
            convlayer = ConvLayer(input_size, kernelsize, output_size, name, **kwargs)
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
    def __init__(self, input_size, conv_params, names, **kwargs):
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
            fclayer = FcLayer(input_size, output_size, name, **kwargs)
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

def create_param(spec, shape, name, trainable=True, regularizable=True):
    if not hasattr(spec, '__call__'):
        assert isinstance(spec, (tf.Tensor, tf.Variable))
        return spec
    assert hasattr(spec, '__call__')
    if regularizable:
        # use the default regularizer
        regularizer = None
    else:
        # do not regularize this variable
        regularizer = lambda _: tf.constant(0.)
    return tf.get_variable(
        name=name, shape=shape, initializer=spec, trainable=trainable,
        regularizer=regularizer, dtype=tf.float32
    )

