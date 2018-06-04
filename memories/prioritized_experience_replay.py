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
import random

import numpy as np

from memories.memory import Memory, Episode, MemoryGranularity, MemoryParameters
from memories.experience_replay import ExperienceReplayParameters, ExperienceReplay
from core_types import Transition
from typing import List, Tuple, Union, Dict, Any
from enum import Enum
import operator
import sys
from schedules import Schedule, ConstantSchedule


class PrioritizedExperienceReplayParameters(ExperienceReplayParameters):
    def __init__(self):
        super().__init__()
        self.max_size = (MemoryGranularity.Transitions, 1000000)
        self.alpha = 0.6
        self.beta = ConstantSchedule(0.4)
        self.epsilon = 1e-6

    @property
    def path(self):
        return 'memories.prioritized_experience_replay:PrioritizedExperienceReplay'


class SegmentTree(object):
    """
    A tree which can be used as a min/max heap or a sum tree
    Add or update item value - O(log N)
    Sampling an item - O(log N)
    """
    class Operation(Enum):
        MAX = {"operator": max, "initial_value": -float("inf")}
        MIN = {"operator": min, "initial_value": 1000}
        SUM = {"operator": operator.add, "initial_value": 0}

    def __init__(self, size: int, operation: Operation):
        self.next_idx_to_write = 0
        self.size = size
        if not (size > 0 and size & (size - 1) == 0):
            raise ValueError("A segment tree size must be a positive power of 2. The given size is {}".format(self.size))
        self.operation = operation
        self.tree = np.ones(2 * size - 1) * self.operation.value['initial_value']
        self.data = [None] * size

    def _propagate(self, idx: int, change: float) -> None:
        """
        Propagate an update of a node's value to its parent node
        :param idx: the index of the node that was updated
        :param change: the change in the value of the node
        :return: None
        """
        parent = (idx - 1) // 2

        self.tree[parent] = self.operation.value['operator'](self.tree[parent], change)

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, val: float)-> int:
        """
        Retrieve the first node that has a value larger than val and is a child of the node at index idx
        :param idx: the index of the root node to search from
        :param val: the value to query for
        :return: the index of the resulting node
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if val <= self.tree[left]:
            return self._retrieve(left, val)
        else:
            return self._retrieve(right, val-self.tree[left])

    def total_value(self) -> float:
        """
        Return the total value of the tree according to the tree operation. For SUM for example, this will return
        the total sum of the tree
        :return: the total value of the tree
        """
        return self.tree[0]

    def add(self, val: float, data: Any) -> None:
        """
        Add a new value to the tree with data assigned to it
        :param val: the new value to add to the tree
        :param data: the data that should be assigned to this value
        :return: None
        """
        idx = self.next_idx_to_write + self.size - 1

        self.data[self.next_idx_to_write] = data
        self.update(idx, val)

        self.next_idx_to_write += 1
        if self.next_idx_to_write >= self.size:
            self.next_idx_to_write = 0

    def update(self, idx: int, new_val: float) -> None:
        """
        Update the value of the node at index idx
        :param idx: the index of the node to update
        :param new_val: the new value of the node
        :return: None
        """
        if not 0 <= idx < len(self.tree):
            raise ValueError("The given index can not be found in the tree")

        change = new_val
        if self.operation == SegmentTree.Operation.SUM:
            change -= self.tree[idx]

        self.tree[idx] = new_val
        self._propagate(idx, change)

    def get(self, val: float) -> Tuple[int, float, Any]:
        """
        Given a value between 0 and the tree sum, return the object which this value is in it's range.
        For example, if we have 3 leaves: 10, 20, 30, and val=35, this will return the 3rd leaf, by accumulating
        leaves by their order until getting to 35. This allows sampling leaves according to their proportional
        probability.
        :param val: a value within the range 0 and the tree sum
        :return: the index of the resulting node in the tree relative to the root node, it's probability and
                 the object itself
        """
        idx = self._retrieve(0, val)
        data_idx = idx - self.size + 1
        data_value = self.tree[idx]
        data = self.data[data_idx]

        return idx, data_value, data

    def __str__(self):
        result = ""
        start = 0
        size = 1
        while size <= self.size:
            result += "{}\n".format(self.tree[start:(start + size)])
            start += size
            size *= 2
        return result


class PrioritizedExperienceReplay(ExperienceReplay):
    """
    This is the proportional sampling variant of the prioritized experience replay as described
    in https://arxiv.org/pdf/1511.05952.pdf.
    """
    def __init__(self, max_size: Tuple[MemoryGranularity, int], alpha: float=0.6, beta: Schedule=ConstantSchedule(0.4),
                 epsilon: float=1e-6):
        """
        :param max_size: the maximum number of transitions or episodes to hold in the memory
        :param alpha: the alpha prioritization coefficient
        :param beta: the beta parameter used for importance sampling
        :param epsilon: a small value added to the priority of each transition
        """
        self.power_of_2_size = 1
        while self.power_of_2_size < max_size[1]:
            self.power_of_2_size *= 2
        super().__init__((MemoryGranularity.Transitions, self.power_of_2_size))
        self.sum_tree = SegmentTree(self.power_of_2_size, SegmentTree.Operation.SUM)
        self.min_tree = SegmentTree(self.power_of_2_size, SegmentTree.Operation.MIN)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.maximal_priority = 1.0

    def update_priority(self, node_idx: int, error: float) -> None:
        """
        Update the priority of a given transition, using its index in the tree and its error
        :param node_idx: the index of the transition leaf in the tree
        :param error: the new error value
        :return: None
        """
        if error <= 0:
            raise ValueError("The priorities must be positive values")
        priority = (error + self.epsilon) ** self.alpha
        self.sum_tree.update(node_idx, priority)
        self.min_tree.update(node_idx, priority)
        self.maximal_priority = max(self.maximal_priority, priority)

    def update_priorities(self, indices: List[int], error_values: List[float]) -> None:
        """
        Update the priorities of a batch of transitions using their indices and their new TD error terms
        :param indices: the indices of the transitions to update
        :param error_values: the new error values
        :return: None
        """
        if len(indices) != len(error_values):
            raise ValueError("The number of indexes requested for update don't match the number of error values given")
        for transition_idx, error in zip(indices, error_values):
            node_idx = transition_idx + self.power_of_2_size - 1  # convert to tree node index
            self.update_priority(node_idx, error)

    def sample(self, size: int) -> List[Transition]:
        """
        Sample a batch of transitions form the replay buffer. If the requested size is larger than the number
        of samples available in the replay buffer then the batch will return empty.
        :param size: the size of the batch to sample
        :param beta: the beta parameter used for importance sampling
        :return: a batch (list) of selected transitions from the replay buffer
        """
        if self.num_transitions() >= size:
            # split the tree leaves to equal segments and sample one transition from each segment
            batch = []
            segment_size = self.sum_tree.total_value() / size

            # get the maximum weight in the memory
            min_probability = self.min_tree.total_value() / self.sum_tree.total_value()
            max_weight = (min_probability * self.num_transitions()) ** -self.beta.current_value

            # sample a batch
            for i in range(size):
                start_probability = segment_size * i
                end_probability = segment_size * (i + 1)

                # sample leaf and calculate its weight
                val = random.uniform(start_probability, end_probability)
                node_idx, priority, transition = self.sum_tree.get(val)
                priority /= self.sum_tree.total_value()
                weight = ((priority * self.num_transitions()) ** -self.beta.current_value) / max_weight

                transition.info['idx'] = node_idx - self.power_of_2_size + 1
                transition.info['weight'] = weight

                batch.append(transition)

            self.beta.step()

            return batch
        else:
            raise ValueError("The replay buffer cannot be sampled since there are not enough transitions yet. "
                             "There are currently {} transitions".format(self.num_transitions()))

    def store(self, transition: Transition) -> None:
        """
        Store a new transition in the memory.
        :param transition: a transition to store
        :return: None
        """
        transition_priority = self.maximal_priority
        self.sum_tree.add(transition_priority, transition)
        self.min_tree.add(transition_priority, transition)
        super().store(transition)

    def clean(self) -> None:
        """
        Clean the memory by removing all the episodes
        :return: None
        """
        super().clean()
        self.sum_tree = SegmentTree(self.power_of_2_size, SegmentTree.Operation.SUM)
        self.min_tree = SegmentTree(self.power_of_2_size, SegmentTree.Operation.MIN)
