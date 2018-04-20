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
from Policy_gen2.qmdp_sublayer import PlannerNet, FilterNet
from sandbox.rocky.tf.core.layers import Layer, MergeLayer

class FilterLayer(Layer):

    def __init__(self, incoming, qmdp_param, **kwargs):

        super(FilterLayer, self).__init__(incoming, **kwargs)

        input_shape = self.input_shape[2:]

        input_dim = np.prod(input_shape)

        # self.qmdp_param = qmdp_param
        self.num_state = qmdp_param['num_state']
        self.num_units = self.num_state
        self.obs_len = qmdp_param['obs_len']
        self.num_action = qmdp_param['num_action']
        self.num_obs = qmdp_param['num_obs']
        # pre-run the step method to initialize the normalization parameters
        # self.build_placeholders()

        self.h0 = self.add_param(tf.constant_initializer(1.0/self.num_state), (self.num_state,), name="h0", trainable=True, regularizable=False)
        self.z_os = self.add_param(tf.constant_initializer(1.0/self.num_obs), (self.num_state*self.num_obs,), name="z_os", trainable=True, regularizable=False)

        self.filternet = FilterNet(qmdp_param, parent_layer=self)
        h_dummy = tf.placeholder(dtype=tf.float32, shape=(None, self.num_units), name="h_dummy")
        x_dummy = tf.placeholder(dtype=tf.float32, shape=(None, input_dim), name="x_dummy")
        self.step(h_dummy, x_dummy)

    def step(self, hprev, x):
        obs_in = x[:,0:self.obs_len]
        act_in = x[:,self.obs_len:]
        prev_b = tf.reshape(hprev,tf.stack([-1,self.num_state]))

        # types conversions
        act_in = tf.to_int32(act_in)
        obs_in = tf.to_float(obs_in)

        n_batches = tf.shape(x)[0]
        z_os= self.z_os
        z_os = tf.tile(tf.reshape(z_os, (1,self.num_state*self.num_obs)),(n_batches,1))
        z_os = tf.to_float(z_os)
        Z = self.filternet.f_Z.step(z_os)

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
        self.num_state = qmdp_param['num_state']
        self.num_units = self.num_state
        self.num_action = qmdp_param['num_action']
        # pre-run the step method to initialize the normalization parameters
        # self.build_placeholders()
        self.R0 = self.add_param(tf.constant_initializer(0.1), (self.num_state*self.num_action,), name="R0", trainable=True, regularizable=False)
        self.V0 = self.add_param(tf.constant_initializer(0.1), (self.num_state), name="V0", trainable=True, regularizable=False)

        self.plannernet = PlannerNet(qmdp_param, parent_layer=self)
        h_dummy = tf.placeholder(dtype=tf.float32, shape=(None, self.num_units), name="h_dummy")
        self.step(h_dummy)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_action)

    def step(self, h):
        b = tf.reshape(h,tf.stack([-1,self.num_state]))

        n_batches = tf.shape(h)[0]
        R0 = self.R0
        R0 = tf.tile(tf.reshape(R0, (1,self.num_state*self.num_action)),(n_batches,1))
        R0 = tf.to_float(R0)

        V0 = self.V0
        V0 = tf.tile(tf.reshape(V0, (1,self.num_state)),(n_batches,1))
        V0 = tf.to_float(V0)

        Q, _, _ = self.plannernet.VI(R0,V0)

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
