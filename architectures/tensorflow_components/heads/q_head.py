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
from core_types import QActionStateValue
from spaces import SpacesDefinition, Box, Discrete
from architectures.tensorflow_components.heads.head import Head


class QHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local)
        self.name = 'q_values_head'
        if isinstance(self.spaces.action, Box):
            self.num_actions = 1
        elif isinstance(self.spaces.action, Discrete):
            self.num_actions = len(self.spaces.action.actions)
        self.return_type = QActionStateValue
        if agent_parameters.network_wrappers[self.network_name].replace_mse_with_huber_loss:
            self.loss_type = tf.losses.huber_loss
        else:
            self.loss_type = tf.losses.mean_squared_error

    def _build_module(self, input_layer):
        # Standard Q Network
        self.output = tf.layers.dense(input_layer, self.num_actions, name='output')
