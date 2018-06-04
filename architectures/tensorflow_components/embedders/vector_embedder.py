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
from typing import List

import tensorflow as tf
from configurations import EmbedderScheme
from architectures.tensorflow_components.embedders.embedder import InputEmbedder
from core_types import InputVectorEmbedding


class VectorEmbedder(InputEmbedder):
    """
    An input embedder that is intended for inputs that can be represented as vectors.
    The embedder flattens the input, applies several dense layers to it and returns the output.
    """
    schemes = {
        EmbedderScheme.Empty:
            [],

        EmbedderScheme.Shallow:
            [
                [128]
            ],

        # dqn
        EmbedderScheme.Medium:
            [
                [256]  # TODO: define this as 128?
            ],

        # carla
        EmbedderScheme.Deep: \
            [
                [128],
                [128],
                [128]
            ]
    }

    def __init__(self, input_size: List[int], activation_function=tf.nn.relu,
                 embedder_scheme: EmbedderScheme=EmbedderScheme.Medium, embedder_width_multiplier: int=1,
                 use_batchnorm: bool=False, use_dropout: bool=False,
                 name: str= "embedder"):
        super().__init__(input_size, activation_function, embedder_scheme, embedder_width_multiplier,
                         use_batchnorm, use_dropout, name)
        self.return_type = InputVectorEmbedding
        self.layers = []
        if len(self.input_size) != 1 and embedder_scheme != EmbedderScheme.Empty:
            raise ValueError("The input size of a vector embedder must contain only a single dimension")

    def _build_module(self):
        # vector observation
        input_layer = tf.contrib.layers.flatten(self.input)
        self.layers.append(input_layer)

        if isinstance(self.embedder_scheme, EmbedderScheme):
            layers_params = self.schemes[self.embedder_scheme]
        else:
            layers_params = self.embedder_scheme
        for idx, layer_params in enumerate(layers_params):
            self.layers.append(
                tf.layers.dense(self.layers[-1], layer_params[0] * self.embedder_width_multiplier,
                                activation=self.activation_function, name='fc{}'.format(idx))
            )

            # batchnorm
            if self.use_batchnorm:
                self.layers.append(
                    tf.layers.batch_normalization(self.layers[-1], name="batchnorm{}".format(idx))
                )

            # activation
            if self.activation_function:
                self.layers.append(
                    self.activation_function(self.layers[-1], name="activation{}".format(idx))
                )

            # dropout
            if self.use_dropout:
                self.layers.append(
                    tf.layers.dropout(self.layers[-1], self.dropout_rate, name="dropout{}".format(idx))
                )

        self.output = self.layers[-1]
