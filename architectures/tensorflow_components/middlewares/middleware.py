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

from core_types import Embedding, MiddlewareEmbedding


class MiddlewareEmbedder(object):
    """
    A middleware embedder is the middle part of the network. It takes the embeddings from the input embedders,
    after they were aggregated in some method (for example, concatenation) and passes it through a neural network
    which can be customizable but shared between the heads of the network
    """
    def __init__(self, activation_function=tf.nn.relu, size=512, name="middleware_embedder"):
        self.name = name
        self.input = None
        self.output = None
        self.activation_function = activation_function
        self.size = size
        self.return_type = MiddlewareEmbedding

    def __call__(self, input_layer):
        with tf.variable_scope(self.get_name()):
            self.input = input_layer
            self._build_module()

        return self.input, self.output

    def _build_module(self):
        pass

    def get_name(self):
        return self.name
