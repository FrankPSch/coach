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

import numpy as np
from typing import Tuple
from enum import Enum
from configurations import Parameters


class MemoryGranularity(Enum):
    Transitions = 0
    Episodes = 1


class MemoryParameters(Parameters):
    def __init__(self):
        super().__init__()
        self.max_size = None
        self.distributed_memory = False
        self.load_memory_from_file_path = None


    @property
    def path(self):
        return 'memories.memory:Memory'


class Memory(object):
    def __init__(self, max_size: Tuple[MemoryGranularity, int]):
        """
        :param max_size: the maximum number of objects to hold in the memory
        """
        self.max_size = max_size
        self._length = 0

    def store(self, obj):
        raise NotImplementedError("")

    def get(self, index):
        raise NotImplementedError("")

    def remove(self, index):
        raise NotImplementedError("")

    def length(self):
        raise NotImplementedError("")

    def sample(self, size):
        raise NotImplementedError("")

    def clean(self):
        raise NotImplementedError("")


class Episode(object):
    def __init__(self):
        self.transitions = []
        # a num_transitions x num_transitions table with the n step return in the n'th row
        self.returns_table = None
        self._length = 0

    def insert(self, transition):
        self.transitions.append(transition)
        self._length += 1

    def is_empty(self):
        return self.length() == 0

    def length(self):
        return self._length

    def get_transition(self, transition_idx):
        return self.transitions[transition_idx]

    def get_last_transition(self):
        return self.get_transition(-1)

    def get_first_transition(self):
        return self.get_transition(0)

    def update_returns(self, discount, is_bootstrapped=False, n_step_return=-1):
        if n_step_return == -1 or n_step_return > self.length():
            n_step_return = self.length()
        rewards = np.array([t.reward for t in self.transitions])
        rewards = rewards.astype('float')
        total_return = rewards.copy()
        current_discount = discount
        for i in range(1, n_step_return):
            total_return += current_discount * np.pad(rewards[i:], (0, i), 'constant', constant_values=0)
            current_discount *= discount

        # calculate the bootstrapped returns
        if is_bootstrapped:
            bootstraps = np.array([np.squeeze(t.info['max_action_value']) for t in self.transitions[n_step_return:]])
            bootstrapped_return = total_return + current_discount * np.pad(bootstraps, (0, n_step_return), 'constant',
                                                                           constant_values=0)
            total_return = bootstrapped_return

        for transition_idx in range(self.length()):
            self.transitions[transition_idx].total_return = total_return[transition_idx]

    def update_measurements_targets(self, num_steps):
        if 'measurements' not in self.transitions[0].state or self.transitions[0].state['measurements'] == []:
            return
        measurements_size = self.transitions[0].state['measurements'].shape[-1]
        total_return = sum([transition.reward for transition in self.transitions])
        for transition_idx, transition in enumerate(self.transitions):
            transition.info['future_measurements'] = np.zeros((num_steps, measurements_size))
            for step in range(num_steps):
                offset_idx = transition_idx + 2 ** step
                if offset_idx >= self.length():
                    offset_idx = -1
                transition.info['future_measurements'][step] = self.transitions[offset_idx].next_state['measurements'] - \
                                                               transition.state['measurements']
            transition.info['total_episode_return'] = total_return

    def update_actions_probabilities(self):
        probability_product = 1
        for transition_idx, transition in enumerate(self.transitions):
            if 'action_probabilities' in transition.info.keys():
                probability_product *= transition.info['action_probabilities']
        for transition_idx, transition in enumerate(self.transitions):
            transition.info['probability_product'] = probability_product

    def get_returns_table(self):
        return self.returns_table

    def get_returns(self):
        return self.get_transitions_attribute('total_return')

    def get_transitions_attribute(self, attribute_name):
        if len(self.transitions) > 0 and hasattr(self.transitions[0], attribute_name):
            return [getattr(t, attribute_name) for t in self.transitions]
        elif len(self.transitions) == 0:
            return []
        else:
            raise ValueError("The transitions have no such attribute name")

    def to_batch(self):
        batch = []
        for i in range(self.length()):
            batch.append(self.get_transition(i))
        return batch
