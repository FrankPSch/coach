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

from exploration_policies.e_greedy import EGreedy, EGreedyParameters
from schedules import Schedule, LinearSchedule, PieceWiseLinearSchedule
from spaces import ActionSpace
import numpy as np
from core_types import RunPhase, ActionType
from typing import List


class UCBParameters(EGreedyParameters):
    def __init__(self):
        super().__init__()
        self.architecture_num_q_heads = 10
        self.bootstrapped_data_sharing_probability = 1.0
        self.epsilon_schedule = PieceWiseLinearSchedule([
            LinearSchedule(1, 0.1, 1000000),
            LinearSchedule(0.1, 0.01, 4000000)
        ])
        self.lamb = 0.1

    @property
    def path(self):
        return 'exploration_policies.ucb:UCB'


class UCB(EGreedy):
    def __init__(self, action_space: ActionSpace, epsilon_schedule: Schedule, evaluation_epsilon: float,
                 noise_percentage_schedule: Schedule, architecture_num_q_heads: int, lamb: int):
        """
        :param action_space: the action space used by the environment
        :param epsilon_schedule: a schedule for the epsilon values
        :param evaluation_epsilon: the epsilon value to use for evaluation phases
        :param noise_percentage_schedule: a schedule for the noise percentage values
        :param architecture_num_q_heads: the number of q heads to select from
        :param lamb: lambda coefficient for taking the standard deviation into account
        """
        super().__init__(action_space, epsilon_schedule, evaluation_epsilon, noise_percentage_schedule)
        self.num_heads = architecture_num_q_heads
        self.lamb = lamb
        self.std = 0
        self.last_action_values = 0

    def select_head(self):
        pass

    def get_action(self, action_values: List[ActionType]) -> ActionType:
        mean = np.mean(action_values, axis=0)
        if self.phase == RunPhase.TRAIN:
            self.std = np.std(action_values, axis=0)
            self.last_action_values = mean + self.lamb * self.std
        else:
            self.last_action_values = mean
        return super().get_action(self.last_action_values)

    def get_control_param(self):
        if self.phase == RunPhase.TRAIN:
            return np.mean(self.std)
        else:
            return 0
