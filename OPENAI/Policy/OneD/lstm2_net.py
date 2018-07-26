import tensorflow as tf
import numpy as np

class lstm(object):
    def __init__(self, name, input_len, nlstm):
        # self.params = params
        self.nlstm = nlstm
        self.f_h = F_h(input_len, nlstm, name)
        self.f_x = F_x(input_len, nlstm, name)

    def update(self, xs, ms, h, c):
        nsteps = len(xs)
        nenv,obs_len = xs[0].get_shape()

        for idx, (x,m) in enumerate(zip(xs,ms)):
            c = c*(1-m)
            h = h*(1-m)
            # z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
            z = self.f_h.step(x) + self.f_x.step(x)
            i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            u = tf.tanh(u)
            c = f*c + i*u
            h = o*tf.tanh(c)
            xs[idx] = h
        s = tf.concat(axis=1, values=[c, h])
        return xs, s



class F_h(object):
    def __init__(self, input_len, nlstm, name):
        w_std = 1.0 / np.sqrt(float(input_len))
        w_mean = 0.0
        dtype = tf.float32
        input_size = input_len
        output_size = nlstm*4
        initializer = tf.truncated_normal_initializer(mean=w_mean, stddev=w_std, dtype=dtype)
        self.wh = create_param(initializer, [input_size, output_size], name=name+'_wh', trainable=True, regularizable=True)
    def step(self, h):
        return tf.matmul(h, self.wh)

class F_x(object):
    def __init__(self, input_len, nlstm, name):
        self.fclayers = FcLayers(input_len, np.array([[nlstm*4, 'lin']]), names=name+"_f_x")
    def step(self, x):
        return self.fclayers.step(x)


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

