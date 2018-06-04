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
from core_types import Embedding
from spaces import SpacesDefinition
from architectures.tensorflow_components.heads.head import Head


# TODO: finish implementing this head
# This is the head used in Feudal Networks
class GoalMappingHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local)
        self.name = 'goal_mapping_head'
        self.goal_pooling = agent_parameters.algorithm.goal_pooling
        self.goal_size = 512  # TODO: this shouldn't be hardcoded
        self.num_actions = self.spaces.action.shape
        self.return_type = Embedding  # TODO - what should be returned here?

        # TODO: predict policy or Q values

    def _build_module(self, input_layer):
        self.goals = tf.placeholder(tf.float32, [None, self.goal_pooling, self.goal_size], name="goals")
        self.input = self.goals

        # goals normalization, pooling and linear projection to w
        # TODO: make sure this does what I intended
        self.normalized_goals = tf.divide(self.goals, tf.norm(self.goals, axis=2), name='normalized_goals')
        self.pooled_goals = tf.reduce_sum(self.normalized_goals, axis=1, name='pooled_goals')
        self.goal_embedding = tf.layers.dense(self.pooled_goals, self.num_actions, use_bias=False, name='goal_projection')

        # combining the goal and action
        self.goal_action_embedding = tf.matmul(input_layer, self.goal_embedding, name='goal_action_embedding')

        # TODO: policy or Q values
        if False:
            self.output = tf.nn.softmax(self.goal_action_embedding, name='policy')
        else:
            self.output = tf.identity(self.goal_action_embedding, name='Q_values')