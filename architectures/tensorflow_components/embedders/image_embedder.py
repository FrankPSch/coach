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
from core_types import InputImageEmbedding


class ImageEmbedder(InputEmbedder):
    """
    An input embedder that performs convolutions on the input and then flattens the result.
    The embedder is intended for image like inputs, where the channels are expected to be the last axis.
    The embedder also allows custom rescaling of the input prior to the neural network.
    """
    schemes = {
        EmbedderScheme.Empty:
            [],

        EmbedderScheme.Shallow:
            [
                [32, 3, 1]
            ],

        # atari dqn
        EmbedderScheme.Medium:
            [
                [32, 8, 4],
                [64, 4, 2],
                [64, 3, 1]
            ],

        # carla
        EmbedderScheme.Deep: \
            [
                [32, 5, 2],
                [32, 3, 1],
                [64, 3, 2],
                [64, 3, 1],
                [128, 3, 2],
                [128, 3, 1],
                [256, 3, 2],
                [256, 3, 1]
            ]
    }

    def __init__(self, input_size: List[int], input_rescaler=255.0, activation_function=tf.nn.relu,
                 embedder_scheme: EmbedderScheme=EmbedderScheme.Medium, embedder_width_multiplier: int=1,
                 use_batchnorm: bool=False, use_dropout: bool=False,
                 name: str= "embedder"):
        super().__init__(input_size, activation_function, embedder_scheme, embedder_width_multiplier,
                         use_batchnorm, use_dropout, name)
        self.input_rescaler = input_rescaler
        self.return_type = InputImageEmbedding
        self.layers = []
        if len(input_size) != 3 and embedder_scheme != EmbedderScheme.Empty:
            raise ValueError("Image embedders expect the input size to have 3 dimensions. The given size is: {}"
                             .format(input_size))

    def _build_module(self):
        # image observation
        rescaled_observation_stack = self.input / self.input_rescaler
        self.layers.append(rescaled_observation_stack)

        # layers order is conv -> batchnorm -> activation -> dropout
        if isinstance(self.embedder_scheme, EmbedderScheme):
            layers_params = self.schemes[self.embedder_scheme]
        else:
            layers_params = self.embedder_scheme
        for idx, layer_params in enumerate(layers_params):
            # convolution
            self.layers.append(
                tf.layers.conv2d(self.layers[-1],
                                 filters=layer_params[0] * self.embedder_width_multiplier,
                                 kernel_size=layer_params[1], strides=layer_params[2],
                                 data_format='channels_last', name='conv{}'.format(idx))
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

        self.output = tf.contrib.layers.flatten(self.layers[-1])


