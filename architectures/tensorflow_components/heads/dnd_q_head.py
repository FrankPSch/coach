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
import multiprocessing
import tensorflow as tf
from configurations import AgentParameters
from spaces import SpacesDefinition
from architectures.tensorflow_components.heads.q_head import QHead


class DNDQHead(QHead):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local)
        self.name = 'dnd_q_values_head'
        self.DND_size = agent_parameters.algorithm.dnd_size
        self.DND_key_error_threshold = agent_parameters.algorithm.DND_key_error_threshold
        self.l2_norm_added_delta = agent_parameters.algorithm.l2_norm_added_delta
        self.new_value_shift_coefficient = agent_parameters.algorithm.new_value_shift_coefficient
        self.number_of_nn = agent_parameters.algorithm.number_of_knn
        self.ap = agent_parameters
        self.dnd_embeddings = [None] * self.num_actions
        self.dnd_values = [None] * self.num_actions
        self.dnd_indices = [None] * self.num_actions
        self.dnd_distances = [None] * self.num_actions
        if self.ap.memory.distributed_memory:
            self.shared_memory_scratchpad = self.ap.task_parameters.shared_memory_scratchpad

    def _build_module(self, input_layer):
        # DND based Q head
        if hasattr(self.ap.task_parameters, 'dnd'):
            self.DND = self.ap.task_parameters.dnd
        else:
            from memories import differentiable_neural_dictionary

            if self.network_parameters.checkpoint_restore_dir:
                self.DND = differentiable_neural_dictionary.load_dnd(self.network_parameters.checkpoint_restore_dir)
            else:
                self.DND = differentiable_neural_dictionary.QDND(
                    self.DND_size, input_layer.get_shape()[-1], self.num_actions, self.new_value_shift_coefficient,
                    key_error_threshold=self.DND_key_error_threshold,
                    learning_rate=self.network_parameters.learning_rate,
                    num_neighbors=self.number_of_nn,
                    override_existing_keys=True)

        # The below code snippet is not in use, as we had an issue with passing the DND through the scratchpad causing
        # multiple errors which result from the use of the Annoy module. Currently, instead, a shared DND has to be
        # initialized and passed externally and directly (not through the scratchpad), as done in the
        # test_custom_actor_critic.

        # my_agent_is_chief = self.ap.task_parameters.task_index == 0
        # lookup_name = self.ap.full_name_id + '.DND'
        # # TODO - currently the DND is not implemented as a memory. while this is broken, we will take abuse of the fact
        # #        that in NEC, we have both a DND and a ER, and so if we decided to have a shared ER, we will also have a
        # #        shared DND
        # if self.ap.memory.distributed_memory is True and not my_agent_is_chief:
        #     self.DND = self.shared_memory_scratchpad.get(lookup_name)
        # else:
        #     if self.ap.memory.distributed_memory:
        #         from memories import protected_differentiable_neural_dictionary as differentiable_neural_dictionary
        #     else:
        #         from memories import differentiable_neural_dictionary
        #
        #     if self.network_parameters.checkpoint_restore_dir:
        #         self.DND = differentiable_neural_dictionary.load_dnd(self.network_parameters.checkpoint_restore_dir)
        #     else:
        #         self.DND = differentiable_neural_dictionary.QDND(
        #             self.DND_size, input_layer.get_shape()[-1], self.num_actions, self.new_value_shift_coefficient,
        #             key_error_threshold=self.DND_key_error_threshold,
        #             learning_rate=self.network_parameters.learning_rate)
        #
        #     if self.ap.memory.distributed_memory is True and my_agent_is_chief:
        #         self.shared_memory_scratchpad.add(lookup_name, self.DND)

        # Retrieve info from DND dictionary
        # We assume that all actions have enough entries in the DND
        self.output = tf.transpose([
            self._q_value(input_layer, action)
            for action in range(self.num_actions)
        ])

    def _q_value(self, input_layer, action):
        result = tf.py_func(self.DND.query,
                            [input_layer, action, self.number_of_nn],
                            [tf.float64, tf.float64, tf.int64])
        self.dnd_embeddings[action] = tf.to_float(result[0])
        self.dnd_values[action] = tf.to_float(result[1])
        self.dnd_indices[action] = result[2]

        # DND calculation
        square_diff = tf.square(self.dnd_embeddings[action] - tf.expand_dims(input_layer, 1))
        distances = tf.reduce_sum(square_diff, axis=2) + [self.l2_norm_added_delta]
        self.dnd_distances[action] = distances
        weights = 1.0 / distances
        normalised_weights = weights / tf.reduce_sum(weights, axis=1, keep_dims=True)
        return tf.reduce_sum(self.dnd_values[action] * normalised_weights, axis=1)

    def _post_build(self):
        # DND gradients
        self.dnd_embeddings_grad = tf.gradients(self.loss[0], self.dnd_embeddings)
        self.dnd_values_grad = tf.gradients(self.loss[0], self.dnd_values)
