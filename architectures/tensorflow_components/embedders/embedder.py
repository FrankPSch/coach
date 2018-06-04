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

from typing import List, Union

import numpy as np
import tensorflow as tf

from configurations import EmbedderScheme
from core_types import InputEmbedding


class InputEmbedder(object):
    """
    An input embedder is the first part of the network, which takes the input from the state and produces a vector
    embedding by passing it through a neural network. The embedder will mostly be input type dependent, and there
    can be multiple embedders in a single network
    """
    def __init__(self, input_size: List[int], activation_function=tf.nn.relu,
                 embedder_scheme: EmbedderScheme=EmbedderScheme.Medium, embedder_width_multiplier: int=1,
                 use_batchnorm: bool=False, use_dropout: bool=False,
                 name: str= "embedder"):
        self.name = name
        self.input_size = input_size
        self.activation_function = activation_function
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.dropout_rate = 0
        self.input = None
        self.output = None
        self.embedder_scheme = embedder_scheme
        self.embedder_width_multiplier = embedder_width_multiplier
        self.return_type = InputEmbedding

    def __call__(self, prev_input_placeholder=None):
        with tf.variable_scope(self.get_name()):
            if prev_input_placeholder is None:
                self.input = tf.placeholder("float", shape=[None] + self.input_size, name=self.get_name())
            else:
                self.input = prev_input_placeholder
            self._build_module()

        return self.input, self.output

    @property
    def input_size(self) -> List[int]:
        return self._input_size

    @input_size.setter
    def input_size(self, value: Union[int, List[int]]):
        if isinstance(value, np.ndarray) or isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, int):
            value = [value]
        if not isinstance(value, list):
            raise ValueError((
                'input_size expected to be a list, found {value} which has type {type}'
            ).format(value=value, type=type(value)))
        self._input_size = value

    def _build_module(self):
        raise NotImplementedError("")

    def get_name(self):
        return self.name