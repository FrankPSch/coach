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

import numpy as np
import tensorflow as tf

from architectures.tensorflow_components.heads.head import Head, normalized_columns_initializer, HeadParameters
from base_parameters import AgentParameters
from core_types import ActionProbabilities
from exploration_policies.continuous_entropy import ContinuousEntropyParameters
from spaces import DiscreteActionSpace, BoxActionSpace
from spaces import SpacesDefinition


class DDPGActorHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='tanh', name: str='policy_head_params'):
        super().__init__(parameterized_class=DDPGActor, activation_function=activation_function, name=name)


class DDPGActor(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='tanh'):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function)
        self.name = 'ddpg_actor_head'
        self.return_type = ActionProbabilities

        self.num_actions = self.spaces.action.shape

        # bounded actions
        self.output_scale = self.spaces.action.max_abs_range

    def _build_module(self, input_layer):
        # mean
        policy_values_mean = tf.layers.dense(input_layer, self.num_actions, name='fc_mean',
                                             activation=self.activation_function)
        self.policy_mean = tf.multiply(policy_values_mean, self.output_scale, name='output_mean')

        self.output = [self.policy_mean]
