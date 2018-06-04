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
from utils import force_list
from configurations import AgentParameters
from spaces import ObservationSpace, ActionSpace, MeasurementsObservationSpace, SpacesDefinition
import numpy as np


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class Head(object):
    """
    A head is the final part of the network. It takes the embedding from the middleware embedder and passes it through
    a neural network to produce the output of the network. There can be multiple heads in a network, and each one has
    an assigned loss function. The heads are algorithm dependent.
    """
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int=0, loss_weight: float=1., is_local: bool=True):
        self.head_idx = head_idx
        self.network_name = network_name
        self.network_parameters = agent_parameters.network_wrappers[self.network_name]
        self.name = "head"
        self.output = []
        self.loss = []
        self.loss_type = []
        self.regularizations = []
        self.loss_weight = force_list(loss_weight)
        self.target = []
        self.importance_weight = []
        self.input = []
        self.is_local = is_local
        self.ap = agent_parameters
        self.spaces = spaces
        self.return_type = None

    def __call__(self, input_layer):
        """
        Wrapper for building the module graph including scoping and loss creation
        :param input_layer: the input to the graph
        :return: the output of the last layer and the target placeholder
        """
        with tf.variable_scope(self.get_name(), initializer=tf.contrib.layers.xavier_initializer()):
            self._build_module(input_layer)

            self.output = force_list(self.output)
            self.target = force_list(self.target)
            self.input = force_list(self.input)
            self.loss_type = force_list(self.loss_type)
            self.loss = force_list(self.loss)
            self.regularizations = force_list(self.regularizations)
            if self.is_local:
                self.set_loss()
            self._post_build()

        if self.is_local:
            return self.output, self.target, self.input, self.importance_weight
        else:
            return self.output, self.input

    def _build_module(self, input_layer):
        """
        Builds the graph of the module
        This method is called early on from __call__. It is expected to store the graph
        in self.output.
        :param input_layer: the input to the graph
        :return: None
        """
        pass

    def _post_build(self):
        """
        Optional function that allows adding any extra definitions after the head has been fully defined
        For example, this allows doing additional calculations that are based on the loss
        :return: None
        """
        pass

    def get_name(self):
        """
        Get a formatted name for the module
        :return: the formatted name
        """
        return '{}_{}'.format(self.name, self.head_idx)

    def set_loss(self):
        """
        Creates a target placeholder and loss function for each loss_type and regularization
        :param loss_type: a tensorflow loss function
        :param scope: the name scope to include the tensors in
        :return: None
        """

        # there are heads that define the loss internally, but we need to create additional placeholders for them
        for loss in self.loss:
            importance_weight = tf.placeholder('float', [None],
                                               '{}_importance_weight'.format(self.get_name()))
            self.importance_weight.append(importance_weight)

        # add losses and target placeholder
        for idx in range(len(self.loss_type)):
            target = tf.placeholder('float', self.output[idx].shape, '{}_target'.format(self.get_name()))
            self.target.append(target)
            importance_weight = tf.placeholder('float', [None],
                                               '{}_importance_weight'.format(self.get_name()))
            self.importance_weight.append(importance_weight)
            loss_weight = tf.expand_dims(self.loss_weight[idx]*importance_weight, axis=-1)
            loss = self.loss_type[idx](self.target[-1], self.output[idx],
                                       weights=loss_weight, scope=self.get_name())
            self.loss.append(loss)

        # add regularizations
        for regularization in self.regularizations:
            self.loss.append(regularization)
