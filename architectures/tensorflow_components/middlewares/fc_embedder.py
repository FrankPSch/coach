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
from core_types import Middleware_FC_Embedding


class FC_Embedder(MiddlewareEmbedder):
    def __init__(self, activation_function=tf.nn.relu, size=512, name="middleware_fc_embedder"):
        super().__init__(activation_function=activation_function, size=size, name=name)
        self.return_type = Middleware_FC_Embedding

    def _build_module(self):
        self.output = tf.layers.dense(self.input, self.size, activation=self.activation_function, name='fc1')
        # self.output = tf.layers.dense(self.input, self.size, activation=tf.nn.tanh, name='fc1')
