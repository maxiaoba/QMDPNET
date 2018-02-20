from tensorpack import graph_builder
import tensorflow as tf
import numpy as np

class PlannerNet():
    @staticmethod
    def f_R(map, goal, num_action, parent_layer=None):
        theta = tf.stack([map, goal], axis=3)
        R = conv_layers(theta, np.array([[3, 150, 'relu'], [1, num_action, 'lin']]), "R_conv", parent_layer=parent_layer)
        return R

    @staticmethod
    def VI(map, goal, params, parent_layer=None):
        """
        builds neural network implementing value iteration. this is the first part of planner module. Fixed through time.
        inputs: map (batch x N x N) and goal(batch)
        returns: Q_K, and optionally: R, list of Q_i
        """
        # build reward model R
        R = PlannerNet.f_R(map, goal, params.num_action,parent_layer=parent_layer)

        # get transition model Tprime. It represents the transition model in the filter, but the weights are not shared.
        kernel = FilterNet.f_T(params.num_action,parent_layer=parent_layer)

        # initialize value image
        V = tf.zeros(map.get_shape().as_list() + [1])
        Q = None

        # repeat value iteration K times
        for i in range(params.K):
            # apply transition and sum
            Q = tf.nn.conv2d(V, kernel, [1, 1, 1, 1], padding='SAME')
            Q = Q + R
            V = tf.reduce_max(Q, axis=[3], keep_dims=True)

        return Q, V, R

    @staticmethod
    def f_pi(q, num_action, parent_layer=None):
        action_pred = fc_layers(q, np.array([[num_action, 'lin']]), names="pi_fc", parent_layer=parent_layer)
        return action_pred

    @staticmethod
    def policy(Q, b, params, reuse=False, parent_layer=None):
        """
        second part of planner module
        :param Q: input Q_K after value iteration
        :param b: belief at current step
        :param params: params
        :return: a_pred,  vector with num_action elements, each has the
        """
        # weight Q by the belief
        b_tiled = tf.tile(tf.expand_dims(b, 3), [1, 1, 1, params.num_action])
        q = tf.multiply(Q, b_tiled)
        # sum over states
        q = tf.reduce_sum(q, [1, 2], keep_dims=False)

        # low-level policy, f_pi
        action_pred = PlannerNet.f_pi(q, params.num_action, parent_layer=parent_layer)

        return action_pred


class FilterNet():
    @staticmethod
    def f_Z(map, params, reuse=False, parent_layer=None):
        """
        This implements f_Z, outputs an observation model (Z). Fixed through time.
        inputs: map (NxN array)
        returns: Z
        """
        # CNN: theta -> Z
        map = tf.expand_dims(map, -1)
        Z = conv_layers(map, np.array([[3, 150, 'lin'], [1, 17, 'sig']]), "Z_conv", parent_layer=parent_layer)

        # normalize over observations
        Z_sum = tf.reduce_sum(Z, [3], keep_dims=True)
        Z = tf.div(Z, Z_sum + 1e-8)  # add a small number to avoid division by zero

        return Z

    @staticmethod
    def f_A(action, num_action, parent_layer=None):
        # identity function
        w_A = tf.one_hot(action, num_action)
        return w_A

    @staticmethod
    def f_O(local_obs, parent_layer=None):

        w_O = fc_layers(local_obs, np.array([[17, 'tanh'], [17, 'smax']]), names="O_fc", parent_layer=parent_layer)
        return w_O

    @staticmethod
    def f_T(num_action, parent_layer=None):
        # get transition kernel
        initializer = tf.truncated_normal_initializer(mean=1.0/9.0, stddev=1.0/90.0, dtype=tf.float32)
        # kernel = tf.get_variable("w_T_conv", [3 * 3, num_action], initializer=initializer, dtype=tf.float32)
        kernel = parent_layer.add_param_plain(initializer, [3 * 3, num_action], name="w_T_conv", trainable=True, regularizable=True)

        # enforce proper probability distribution (i.e. values must sum to one) by softmax
        kernel = tf.nn.softmax(kernel, dim=0)
        kernel = tf.reshape(kernel, [3, 3, 1, num_action], name="T_w")

        return kernel

    @staticmethod
    def beliefupdate(Z, b, action, local_obs, params, reuse=False, parent_layer=None):
        """
        Belief update in the filter module with pre-computed Z.
        :param b: belief (b_i), [batch, N, M, 1]
        :param action: action input (a_i)
        :param obs: observation input (o_i)
        :return: updated belief b_(i+1)
        """
        # step 1: update belief with transition
        # get transition kernel (T)
        kernel = FilterNet.f_T(params.num_action, parent_layer=parent_layer)

        # apply convolution which corresponds to the transition function in an MDP (f_T)
        b = tf.expand_dims(b, -1)
        b_prime = tf.nn.conv2d(b, kernel, [1, 1, 1, 1], padding='SAME')

        # index into the appropriate channel of b_prime
        w_A = FilterNet.f_A(action, params.num_action, parent_layer=parent_layer)
        w_A = w_A[:, None, None]
        b_prime_a = tf.reduce_sum(tf.multiply(b_prime, w_A), [3], keep_dims=False) # soft indexing

        #b_prime_a = tf.abs(b_prime_a) # TODO there was this line. does it make a difference with softmax?

        # step 2: update belief with observation
        # get observation probabilities for the obseravtion input by soft indexing
        w_O = FilterNet.f_O(local_obs, parent_layer=parent_layer)
        w_O = w_O[:,None,None] #tf.expand_dims(tf.expand_dims(w_O, axis=1), axis=1)
        Z_o = tf.reduce_sum(tf.multiply(Z, w_O), [3], keep_dims=False) # soft indexing

        b_next = tf.multiply(b_prime_a, Z_o)

        # step 3: normalize over the state space
        # add small number to avoid division by zero
        b_next = tf.div(b_next, tf.reduce_sum(b_next, [1, 2], keep_dims=True) + 1e-8)

        return b_next


# Helper function to construct layers conveniently

def conv_layer(input, kernel_size, filters, name, w_mean=0.0, w_std=None, addbias=True, strides=(1, 1, 1, 1), padding='SAME', parent_layer=None):
    """
    Create variables and operator for a convolutional layer
    :param input: input tensor
    :param kernel_size: size of kernel
    :param filters: number of convolutional filters
    :param name: variable name for convolutional kernel and bias
    :param w_mean: mean of initializer for kernel weights
    :param w_std: standard deviation of initializer for kernel weights. Use 1/sqrt(input_param_count) if None.
    :param addbias: add bias if True
    :param strides: convolutional strides, match TF
    :param padding: padding, match TF
    :return: output tensor
    """
    dtype = tf.float32

    input_size = int(input.get_shape()[3], )
    if w_std is None:
        w_std = 1.0 / np.sqrt(float(input_size * kernel_size * kernel_size))

    # kernel = tf.get_variable('w_'+name,
    #                          [kernel_size, kernel_size, input_size, filters],
    #                          initializer=tf.truncated_normal_initializer(mean=w_mean, stddev=w_std, dtype=dtype),
    #                          dtype=dtype)

    initializer = tf.truncated_normal_initializer(mean=w_mean, stddev=w_std, dtype=dtype)    
    kernel = parent_layer.add_param_plain(initializer, [kernel_size, kernel_size, input_size, filters], name='w_'+name, trainable=True, regularizable=True)

    output = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)

    if addbias:
        # biases = tf.get_variable('b_' + name, [filters], initializer=tf.constant_initializer(0.0))
        initializer = tf.constant_initializer(0.0)
        biases = parent_layer.add_param_plain(initializer, [filters], name='b_' + name, trainable=True, regularizable=True)
        output = tf.nn.bias_add(output, biases)
    return output


def linear_layer(input, output_size, name, w_mean=0.0, w_std=None, parent_layer=None,):
    """
    Create variables and operator for a linear layer
    :param input: input tensor
    :param output_size: output size, number of hidden units
    :param name: variable name for linear weights and bias
    :param w_mean: mean of initializer for kernel weights
    :param w_std: standard deviation of initializer for kernel weights. Use 1/sqrt(input_param_count) if None.
    :return: output tensor
    """
    dtype = tf.float32

    if w_std is None:
        w_std = 1.0 / np.sqrt(float(np.prod(input.get_shape().as_list()[1])))

    # w = tf.get_variable('w_' + name,
    #                     [input.get_shape()[1], output_size],
    #                     initializer=tf.truncated_normal_initializer(mean=w_mean, stddev=w_std, dtype=dtype),
    #                     dtype=dtype)

    initializer = tf.truncated_normal_initializer(mean=w_mean, stddev=w_std, dtype=dtype)
    w = parent_layer.add_param_plain(initializer, [input.get_shape()[1], output_size], name='w_' + name, trainable=True, regularizable=True)

    # b = tf.get_variable("b_" + name, [output_size], initializer=tf.constant_initializer(0.0))

    initializer = tf.constant_initializer(0.0)
    b = parent_layer.add_param_plain(initializer, [output_size], name='b_' + name, trainable=True, regularizable=True)

    output = tf.matmul(input, w) + b

    return output


def conv_layers(input, conv_params, names, parent_layer=None, **kwargs):
    """ Build convolution layers from a list of descriptions.
        Each descriptor is a list: [kernel, hidden filters, activation]
    """
    output = input
    for layer_i in range(conv_params.shape[0]):
        kernelsize = int(conv_params[layer_i][0])
        hiddensize = int(conv_params[layer_i][1])
        if isinstance(names, list):
            name = names[layer_i]
        else:
            name = names+'_%d'%layer_i
        output = conv_layer(output, kernelsize, hiddensize, name, parent_layer=parent_layer, **kwargs)
        output = activation(output, conv_params[layer_i][2])
    return output


def fc_layers(input, conv_params, names, parent_layer=None, **kwargs):
    """ Build convolution layers from a list of descriptions.
        Each descriptor is a list: [size, _, activation]
    """
    output = input
    for layer_i in range(conv_params.shape[0]):
        size = int(conv_params[layer_i][0])
        if isinstance(names, list):
            name = names[layer_i]
        else:
            name = names+'_%d'%layer_i
        output = linear_layer(output, size, name, parent_layer=parent_layer, **kwargs)
        output = activation(output, conv_params[layer_i][-1])
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