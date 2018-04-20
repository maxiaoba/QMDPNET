import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
import numpy as np
import itertools
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.parameterized import Parameterized
from sandbox.rocky.tf.core.layers_powered import LayersPowered
import pickle, argparse, os
from utils.dotdict import dotdict
from Policy_gen.qmdp_layer import FilterLayer, PlannerLayer
import joblib

class QMDPNetwork(object):
    def __init__(self, name, input_shape, output_dim, input_var=None, input_layer=None, qmdp_param=None):
        with tf.variable_scope(name):
            hidden_dim = qmdp_param['num_state']
            if input_layer is None:
                l_in = L.InputLayer(shape=(None, None) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer

            l_step_input = L.InputLayer(shape=(None,) + input_shape, name="step_input")
            l_step_prev_state = L.InputLayer(shape=(None, hidden_dim), name="step_prev_state")

            l_gru = FilterLayer(l_in, qmdp_param, name="qmdp_filter")

            l_gru_flat = L.ReshapeLayer(
                l_gru, shape=(-1, hidden_dim),
                name="gru_flat"
            )
            l_output_flat = PlannerLayer(
                l_gru_flat,
                qmdp_param,
                name="output_flat"
            )

            l_output = L.OpLayer(
                l_output_flat,
                op=lambda flat_output, l_input:
                tf.reshape(flat_output, tf.stack((tf.shape(l_input)[0], tf.shape(l_input)[1], -1))),
                shape_op=lambda flat_output_shape, l_input_shape:
                (l_input_shape[0], l_input_shape[1], flat_output_shape[-1]),
                extras=[l_in],
                name="output"
            )
            l_step_state = l_gru.get_step_layer(l_step_input, l_step_prev_state, name="step_state")
            l_step_hidden = l_step_state
            l_step_output = l_output_flat.get_step_layer(l_step_hidden, name="step_output")

            self._l_in = l_in
            self._hid_init_param = l_gru.h0
            self._l_gru = l_gru
            self._l_output_flat = l_output_flat
            self._l_out = l_output
            self._l_step_input = l_step_input
            self._l_step_prev_state = l_step_prev_state
            self._l_step_hidden = l_step_hidden
            self._l_step_state = l_step_state
            self._l_step_output = l_step_output
            self._hidden_dim = hidden_dim

    @property
    def state_dim(self):
        return self._hidden_dim

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @property
    def input_layer(self):
        return self._l_in

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def output_layer(self):
        return self._l_out

    @property
    def recurrent_layer(self):
        return self._l_gru

    @property
    def step_input_layer(self):
        return self._l_step_input

    @property
    def step_prev_state_layer(self):
        return self._l_step_prev_state

    @property
    def step_hidden_layer(self):
        return self._l_step_hidden

    @property
    def step_state_layer(self):
        return self._l_step_state

    @property
    def step_output_layer(self):
        return self._l_step_output

    @property
    def hid_init_param(self):
        return self._hid_init_param

    @property
    def state_init_param(self):
        return self._hid_init_param