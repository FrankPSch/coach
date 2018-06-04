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
from core_types import ActionProbabilities
from spaces import SpacesDefinition
from architectures.tensorflow_components.heads.head import Head, normalized_columns_initializer
import numpy as np
from spaces import Discrete, Box
from exploration_policies.continuous_entropy import ContinuousEntropy, ContinuousEntropyParameters


class PolicyHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local)
        self.name = 'policy_values_head'
        self.return_type = ActionProbabilities

        self.exploration_policy = agent_parameters.exploration
        if isinstance(self.spaces.action, Discrete):
            self.num_actions = len(self.spaces.action.actions)
        elif isinstance(self.spaces.action, Box):
            self.num_actions = self.spaces.action.shape
            if np.all(self.spaces.action.max_abs_range < np.inf):
                # bounded actions
                self.output_scale = self.spaces.action.max_abs_range
                self.continuous_output_activation = tf.nn.tanh

            else:
                # unbounded actions
                self.output_scale = 1
                self.continuous_output_activation = None

        if hasattr(agent_parameters.algorithm, 'beta_entropy'):
            self.beta = agent_parameters.algorithm.beta_entropy
        else:
            self.beta = None

    def _build_module(self, input_layer):
        eps = 1e-15
        if isinstance(self.spaces.action, Discrete):
            self.actions = tf.placeholder(tf.int32, [None], name="actions")
        elif isinstance(self.spaces.action, Box):
            self.actions = tf.placeholder(tf.float32, [None, self.num_actions], name="actions")
        else:
            raise ValueError("Only discrete or continuous action spaces are supported for the policy head")
        self.input = [self.actions]

        # Policy Head
        if isinstance(self.spaces.action, Discrete):
            policy_values = tf.layers.dense(input_layer, self.num_actions, name='fc')
            self.policy_mean = tf.nn.softmax(policy_values, name="policy")

            # define the distributions for the policy and the old policy
            # (the + eps is to prevent probability 0 which will cause the log later on to be -inf)
            self.policy_distribution = tf.contrib.distributions.Categorical(probs=(self.policy_mean + eps))
            self.output = self.policy_mean
        elif isinstance(self.spaces.action, Box):
            # mean
            policy_values_mean = tf.layers.dense(input_layer, self.num_actions, activation=tf.nn.tanh, name='fc_mean')
            self.policy_mean = tf.multiply(policy_values_mean, self.output_scale, name='output_mean')

            self.output = [self.policy_mean]

            # std
            if isinstance(self.exploration_policy, ContinuousEntropyParameters):
                policy_values_std = tf.layers.dense(input_layer, self.num_actions,
                                            kernel_initializer=normalized_columns_initializer(0.01), name='fc_std')
                self.policy_std = tf.nn.softplus(policy_values_std, name='output_variance') + eps

                self.output.append(self.policy_std)

            else:
                self.policy_std = tf.Variable(np.ones(self.num_actions), dtype='float32', trainable=False)

                # assign op for the policy std
                self.policy_std_placeholder = tf.placeholder('float32', (self.num_actions,))
                self.assign_policy_std = tf.assign(self.policy_std, self.policy_std_placeholder)

            # define the distributions for the policy and the old policy
            self.policy_distribution = tf.contrib.distributions.MultivariateNormalDiag(self.policy_mean,
                                                                                       self.policy_std)

        if self.is_local:
            # add entropy regularization
            if self.beta:
                self.entropy = tf.reduce_mean(self.policy_distribution.entropy())
                self.regularizations = -tf.multiply(self.beta, self.entropy, name='entropy_regularization')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.regularizations)

            # calculate loss
            self.action_log_probs_wrt_policy = self.policy_distribution.log_prob(self.actions)
            self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
            self.target = self.advantages
            self.loss = -tf.reduce_mean(self.action_log_probs_wrt_policy * self.advantages)
            tf.losses.add_loss(self.loss_weight[0] * self.loss)
