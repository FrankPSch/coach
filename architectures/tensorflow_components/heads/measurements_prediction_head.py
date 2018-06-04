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
from configurations import AgentParameters
from core_types import Measurements
from spaces import SpacesDefinition
from architectures.tensorflow_components.heads.head import Head


class MeasurementsPredictionHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local)
        self.name = 'future_measurements_head'
        self.num_actions = self.spaces.action.shape
        self.num_measurements = self.spaces.measurements.shape
        self.num_prediction_steps = agent_parameters.algorithm.num_predicted_steps_ahead
        self.multi_step_measurements_size = self.num_measurements * self.num_prediction_steps
        self.return_type = Measurements
        if agent_parameters.network_wrappers[self.network_name].replace_mse_with_huber_loss:
            self.loss_type = tf.losses.huber_loss
        else:
            self.loss_type = tf.losses.mean_squared_error

    def _build_module(self, input_layer):
        # This is almost exactly the same as Dueling Network but we predict the future measurements for each action
        # actions expectation tower (expectation stream) - E
        with tf.variable_scope("expectation_stream"):
            expectation_stream = tf.layers.dense(input_layer, 256, activation=tf.nn.elu, name='fc1')
            expectation_stream = tf.layers.dense(expectation_stream, self.multi_step_measurements_size, name='output')
            expectation_stream = tf.expand_dims(expectation_stream, axis=1)

        # action fine differences tower (action stream) - A
        with tf.variable_scope("action_stream"):
            action_stream = tf.layers.dense(input_layer, 256, activation=tf.nn.elu, name='fc1')
            action_stream = tf.layers.dense(action_stream, self.num_actions * self.multi_step_measurements_size,
                                            name='output')
            action_stream = tf.reshape(action_stream,
                                       (tf.shape(action_stream)[0], self.num_actions, self.multi_step_measurements_size))
            action_stream = action_stream - tf.reduce_mean(action_stream, reduction_indices=1, keep_dims=True)

        # merge to future measurements predictions
        self.output = tf.add(expectation_stream, action_stream, name='output')
