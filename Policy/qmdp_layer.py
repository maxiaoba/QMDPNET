import functools
import numpy as np
import math
import tensorflow as tf
from tensorflow.python.training import moving_averages
from collections import OrderedDict
from collections import deque
from itertools import chain
from inspect import getargspec
from difflib import get_close_matches
from warnings import warn
from qmdp_sublayer import PlannerNet, FilterNet
from sandbox.rocky.tf.core.layers import Layer, MergeLayer

class FilterLayer(Layer):

    def __init__(self, incoming, qmdp_param, **kwargs):

        super(FilterLayer, self).__init__(incoming, **kwargs)

        input_shape = self.input_shape[2:]

        input_dim = np.prod(input_shape)

        # self.qmdp_param = qmdp_param
        self.N = qmdp_param['grid_n']
        self.M = qmdp_param['grid_m']
        self.obs_len = qmdp_param['obs_len']
        self.num_units = self.N*self.M
        # pre-run the step method to initialize the normalization parameters
        # self.build_placeholders()
        self.map = self.add_param(tf.constant_initializer(0.0), (self.N, self.M), name="map", trainable=False, regularizable=False)
        self.goal = self.add_param(tf.constant_initializer(0.0), (self.N, self.M), name="goal", trainable=False, regularizable=False)                    
        self.h0 = self.add_param(tf.constant_initializer(0.0), (self.num_units,), name="h0", trainable=False, regularizable=False)
        self.filternet = FilterNet(qmdp_param, parent_layer=self)
        h_dummy = tf.placeholder(dtype=tf.float32, shape=(None, self.num_units), name="h_dummy")
        x_dummy = tf.placeholder(dtype=tf.float32, shape=(None, input_dim), name="x_dummy")
        self.step(h_dummy, x_dummy)

    def step(self, hprev, x):
        # map, goal = self.placeholders
        n_batches = tf.shape(hprev)[0]
        map, goal = self.map, self.goal
        map = tf.tile(
                tf.reshape(map, (1, self.N, self.M)),
                (n_batches, 1, 1))
        goal = tf.tile(
                tf.reshape(goal, (1, self.N, self.M)),
                (n_batches, 1, 1))

        obs_in = x[:,0:self.obs_len]
        act_in = x[:,self.obs_len:]
        prev_b = tf.reshape(hprev,tf.stack([-1,self.N,self.M]))

        # types conversions
        map = tf.to_float(map)
        goal = tf.to_float(goal)
        act_in = tf.to_int32(act_in)
        obs_in = tf.to_float(obs_in)

        with tf.variable_scope("filter"):
            # Z = FilterNet.f_Z(map, self.qmdp_param, parent_layer=self)
            Z = self.filternet.f_Z.step(map)

        # create variable for hidden belief (equivalent to the hidden state of an RNN)
        # self.belief = tf.Variable(np.zeros(prev_b.get_shape().as_list(), 'f'), trainable=False, name="hidden_belief")

        # filter
        with tf.variable_scope("filter") as step_scope:
            # self.b = FilterNet.beliefupdate(Z, prev_b, act_in, obs_in, self.qmdp_param, parent_layer=self)
            b = self.filternet.beliefupdate(Z, prev_b, act_in, obs_in)

        h = tf.reshape(b,tf.stack([-1,self.num_units]))
        return h


    def get_step_layer(self, l_in, l_prev_hidden, name=None):
        return FilterStepLayer(incomings=[l_in, l_prev_hidden], filter_layer=self, name=name)

    def get_output_shape_for(self, input_shape):
        n_batch, n_steps = input_shape[:2]
        return n_batch, n_steps, self.num_units

    def get_output_for(self, input, **kwargs):
        input_shape = tf.shape(input)
        n_batches = input_shape[0]
        n_steps = input_shape[1]
        input = tf.reshape(input, tf.stack([n_batches, n_steps, -1]))
        if 'recurrent_state' in kwargs and self in kwargs['recurrent_state']:
            h0s = kwargs['recurrent_state'][self]
        else:
            # use this
            h0s = tf.tile(
                tf.reshape(self.h0, (1, self.num_units)),
                (n_batches, 1)
            )
        # h0s = tf.tile(
        #     tf.reshape(self.h0, (1, self.num_units)),
        #     (n_batches, 1)
        # )
        # flatten extra dimensions
        shuffled_input = tf.transpose(input, (1, 0, 2))
        hs = tf.scan(
            self.step,
            elems=shuffled_input,
            initializer=h0s
        )
        shuffled_hs = tf.transpose(hs, (1, 0, 2))
        if 'recurrent_state_output' in kwargs:
            kwargs['recurrent_state_output'][self] = shuffled_hs
        return shuffled_hs


class FilterStepLayer(MergeLayer):
    def __init__(self, incomings, filter_layer, **kwargs):
        super(FilterStepLayer, self).__init__(incomings, **kwargs)
        self._filter_layer = filter_layer

    def get_params(self, **tags):
        return self._filter_layer.get_params(**tags)

    def get_output_shape_for(self, input_shapes):
        n_batch = input_shapes[0][0]
        return n_batch, self._filter_layer.num_units

    def get_output_for(self, inputs, **kwargs):
        x, hprev = inputs
        n_batch = tf.shape(x)[0]
        x = tf.reshape(x, tf.stack([n_batch, -1]))
        x.set_shape((None, self.input_shapes[0][1]))
        return self._filter_layer.step(hprev, x)

class PlannerLayer(Layer):
    def __init__(self, incoming, qmdp_param, **kwargs):

        super(PlannerLayer, self).__init__(incoming, **kwargs)

        input_shape = self.input_shape[2:]

        input_dim = np.prod(input_shape)

        # self.qmdp_param = qmdp_param
        self.N = qmdp_param['grid_n']
        self.M = qmdp_param['grid_m']
        self.num_units = self.N*self.M
        self.num_action = qmdp_param['num_action']
        # pre-run the step method to initialize the normalization parameters
        # self.build_placeholders()
        self.map = self.add_param(tf.constant_initializer(0.0), (self.N, self.M), name="map", trainable=False, regularizable=False)
        self.goal = self.add_param(tf.constant_initializer(0.0), (self.N, self.M), name="goal", trainable=False, regularizable=False)                    
        self.h0 = self.add_param(tf.constant_initializer(0.0), (self.num_units,), name="h0", trainable=False, regularizable=False)
        self.plannernet = PlannerNet(qmdp_param, parent_layer=self)
        h_dummy = tf.placeholder(dtype=tf.float32, shape=(None, self.num_units), name="h_dummy")
        self.step(h_dummy)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_action)

    def step(self, h):
        b = tf.reshape(h,tf.stack([-1,self.N,self.M]))

        n_batches = tf.shape(h)[0]
        map, goal = self.map, self.goal
        map = tf.tile(
                tf.reshape(map, (1, self.N, self.M)),
                (n_batches, 1, 1))
        goal = tf.tile(
                tf.reshape(goal, (1, self.N, self.M)),
                (n_batches, 1, 1))

        # types conversions
        map = tf.to_float(map)
        goal = tf.to_float(goal)

        with tf.variable_scope("planner"):
            # Q, _, _ = PlannerNet.VI(map, goal, self.qmdp_param, parent_layer=self)
            Q, _, _ = self.plannernet.VI(map, goal)
        with tf.variable_scope("planner") as step_scope:
            # action_pred = PlannerNet.policy(Q, b, self.qmdp_param, parent_layer=self)
            action_pred = self.plannernet.policy(Q, b)

        return action_pred

    def get_output_for(self, input, **kwargs):
        if input.get_shape().ndims > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = tf.reshape(input, tf.stack([tf.shape(input)[0], -1]))
        return self.step(input)

    def get_step_layer(self, l_in, name=None):
        return PlannerStepLayer(incomings=[l_in], planner_layer=self, name=name)

class PlannerStepLayer(MergeLayer):
    def __init__(self, incomings, planner_layer, **kwargs):
        super(PlannerStepLayer, self).__init__(incomings, **kwargs)
        self._planner_layer = planner_layer

    def get_params(self, **tags):
        return self._planner_layer.get_params(**tags)

    def get_output_shape_for(self, input_shapes):
        n_batch = input_shapes[0][0]
        return n_batch, self._planner_layer.num_action

    def get_output_for(self, inputs, **kwargs):
        return self._planner_layer.get_output_for(inputs[0])
