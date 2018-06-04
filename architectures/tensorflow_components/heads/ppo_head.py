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
from architectures.tensorflow_components.heads.head import Head
import numpy as np
from spaces import Box, Discrete
from utils import eps


class PPOHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local)
        self.name = 'ppo_head'
        self.num_actions = self.spaces.action.shape[-1]
        self.return_type = ActionProbabilities
        if isinstance(self.spaces.action, Box):
            self.output_scale = self.spaces.action.max_abs_range

        self.use_kl_regularization = agent_parameters.algorithm.use_kl_regularization
        if self.use_kl_regularization:
            # kl coefficient and its corresponding assignment operation and placeholder
            self.kl_coefficient = tf.Variable(agent_parameters.algorithm.initial_kl_coefficient,
                                              trainable=False, name='kl_coefficient')
            self.kl_coefficient_ph = tf.placeholder('float', name='kl_coefficient_ph')
            self.assign_kl_coefficient = tf.assign(self.kl_coefficient, self.kl_coefficient_ph)
            self.kl_cutoff = 2 * agent_parameters.algorithm.target_kl_divergence
            self.high_kl_penalty_coefficient = agent_parameters.algorithm.high_kl_penalty_coefficient
        self.clip_likelihood_ratio_using_epsilon = agent_parameters.algorithm.clip_likelihood_ratio_using_epsilon
        self.beta = agent_parameters.algorithm.beta_entropy

    def _build_module(self, input_layer):
        if isinstance(self.spaces.action, Discrete):
            self.actions = tf.placeholder(tf.int32, [None], name="actions")
        elif isinstance(self.spaces.action, Box):
            self.actions = tf.placeholder(tf.float32, [None, self.num_actions], name="actions")
        else:
            raise ValueError("only discrete or continuous action spaces are supported for PPO")
        self.old_policy_mean = tf.placeholder(tf.float32, [None, self.num_actions], "old_policy_mean")
        self.old_policy_std = tf.placeholder(tf.float32, [None, self.num_actions], "old_policy_std")

        # Policy Head
        if isinstance(self.spaces.action, Discrete):
            self.input = [self.actions, self.old_policy_mean]
            policy_values = tf.layers.dense(input_layer, self.num_actions, name='policy_fc')
            self.policy_mean = tf.nn.softmax(policy_values, name="policy")

            # define the distributions for the policy and the old policy
            self.policy_distribution = tf.contrib.distributions.Categorical(probs=self.policy_mean)
            self.old_policy_distribution = tf.contrib.distributions.Categorical(probs=self.old_policy_mean)

            self.output = self.policy_mean
        elif isinstance(self.spaces.action, Box):
            self.input = [self.actions, self.old_policy_mean, self.old_policy_std]
            self.policy_mean = tf.layers.dense(input_layer, self.num_actions, name='policy_mean')
            self.policy_logstd = tf.Variable(np.zeros((1, self.num_actions)), dtype='float32')
            self.policy_std = tf.tile(tf.exp(self.policy_logstd), [tf.shape(input_layer)[0], 1], name='policy_std')

            # define the distributions for the policy and the old policy
            self.policy_distribution = tf.distributions.Normal(self.policy_mean,
                                                                                       self.policy_std + eps)
            self.old_policy_distribution = tf.distributions.Normal(self.old_policy_mean,
                                                                                           self.old_policy_std + eps)

            self.output = [self.policy_mean, self.policy_std]

        self.action_probs_wrt_policy = tf.exp(self.policy_distribution.log_prob(self.actions))
        self.action_probs_wrt_old_policy = tf.exp(self.old_policy_distribution.log_prob(self.actions))
        self.entropy = tf.reduce_mean(self.policy_distribution.entropy())

        # add kl divergence regularization  # TODO: the kl divergence does not work in TF 1.4.1. either switch to explicit calculation or switch TF
        self.kl_divergence = tf.reduce_mean(tf.distributions.kl_divergence(self.old_policy_distribution, self.policy_distribution))
        # self.kl_divergence = tf.reduce_mean(tf.log(self.old_policy_std / self.policy_std) +
        #                                     (self.policy_std**2 + (self.old_policy_mean- self.policy_mean)**2) /
        #                                     (2*self.old_policy_std**2) - 1/2)
        if self.use_kl_regularization:
            # no clipping => use kl regularization
            self.weighted_kl_divergence = tf.multiply(self.kl_coefficient, self.kl_divergence)
            self.regularizations = self.weighted_kl_divergence + self.high_kl_penalty_coefficient * \
                                                tf.square(tf.maximum(0.0, self.kl_divergence - self.kl_cutoff))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.regularizations)

        # calculate surrogate loss
        self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
        self.target = self.advantages
        # action_probs_wrt_old_policy != 0 because it is e^...
        self.likelihood_ratio = self.action_probs_wrt_policy / (self.action_probs_wrt_old_policy + eps)
        if self.clip_likelihood_ratio_using_epsilon is not None:
            max_value = 1 + self.clip_likelihood_ratio_using_epsilon
            min_value = 1 - self.clip_likelihood_ratio_using_epsilon
            self.clipped_likelihood_ratio = tf.clip_by_value(self.likelihood_ratio, min_value, max_value)
            self.scaled_advantages = tf.minimum(self.likelihood_ratio * self.advantages,
                                                self.clipped_likelihood_ratio * self.advantages)
        else:
            self.scaled_advantages = self.likelihood_ratio * self.advantages
        # minus sign is in order to set an objective to minimize (we actually strive for maximizing the surrogate loss)
        self.surrogate_loss = -tf.reduce_mean(self.scaled_advantages)
        if self.is_local:
            # add entropy regularization
            if self.beta:
                self.entropy = tf.reduce_mean(self.policy_distribution.entropy())
                self.regularizations = -tf.multiply(self.beta, self.entropy, name='entropy_regularization')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.regularizations)

        self.loss = self.surrogate_loss
        tf.losses.add_loss(self.loss)
