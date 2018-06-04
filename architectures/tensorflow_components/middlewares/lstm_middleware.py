#
# Copyright (c) 2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import tensorflow as tf
from architectures.tensorflow_components.middlewares.middleware import MiddlewareEmbedder
import numpy as np

from core_types import Middleware_LSTM_Embedding


class LSTM_Embedder(MiddlewareEmbedder):
    def __init__(self, activation_function=tf.nn.relu, size=512, name="middleware_lstm_embedder"):
        super().__init__(activation_function=activation_function, size=size, name=name)
        self.return_type = Middleware_LSTM_Embedding

    def _build_module(self):
        """
        self.state_in: tuple of placeholders containing the initial state
        self.state_out: tuple of output state

        todo: it appears that the shape of the output is batch, feature
        the code here seems to be slicing off the first element in the batch
        which would definitely be wrong. need to double check the shape
        """

        middleware = tf.layers.dense(self.input, self.size, activation=self.activation_function, name='fc1')
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
        self.c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        self.h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        self.state_init = [self.c_init, self.h_init]
        self.c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        self.h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        self.state_in = (self.c_in, self.h_in)
        rnn_in = tf.expand_dims(middleware, [0])
        step_size = tf.shape(middleware)[:1]
        state_in = tf.contrib.rnn.LSTMStateTuple(self.c_in, self.h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size, time_major=False)
        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
        self.output = tf.reshape(lstm_outputs, [-1, 256])
