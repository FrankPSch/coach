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

from exploration_policies.exploration_policy import ExplorationPolicy
from schedules import Schedule, LinearSchedule
from spaces import ActionSpace, Discrete, Box
import numpy as np
from core_types import RunPhase, ActionType
from typing import List
from exploration_policies.exploration_policy import ExplorationParameters


class EGreedyParameters(ExplorationParameters):
    def __init__(self):
        super().__init__()
        self.epsilon_schedule = LinearSchedule(0.5, 0.01, 50000)
        self.evaluation_epsilon = 0.05
        self.noise_percentage_schedule = LinearSchedule(0.1, 0.1, 50000) # for continuous control -
        # (see http://www.cs.ubc.ca/~van/papers/2017-TOG-deepLoco/2017-TOG-deepLoco.pdf)

    @property
    def path(self):
        return 'exploration_policies.e_greedy:EGreedy'


class EGreedy(ExplorationPolicy):
    def __init__(self, action_space: ActionSpace, epsilon_schedule: Schedule,
                 evaluation_epsilon: float, noise_percentage_schedule: Schedule=None):
        """
        :param action_space: the action space used by the environment
        :param epsilon_schedule: a schedule for the epsilon values
        :param evaluation_epsilon: the epsilon value to use for evaluation phases
        :param noise_percentage_schedule: a schedule for the noise percentage values
        """
        ExplorationPolicy.__init__(self, action_space)
        self.epsilon_schedule = epsilon_schedule
        self.evaluation_epsilon = evaluation_epsilon
        # for continuous e-greedy (see http://www.cs.ubc.ca/~van/papers/2017-TOG-deepLoco/2017-TOG-deepLoco.pdf)
        self.variance_schedule = noise_percentage_schedule

        if type(action_space) == Box and noise_percentage_schedule is None:
            raise ValueError("For continuous controls, the noise schedule should be supplied to the exploration policy")

    def get_action(self, action_values: List[ActionType]) -> ActionType:
        if self.phase == RunPhase.TRAIN:
            self.epsilon_schedule.step()
            if self.variance_schedule:
                self.variance_schedule.step()
        epsilon = self.evaluation_epsilon if self.phase == RunPhase.TEST else self.epsilon_schedule.current_value

        if isinstance(self.action_space, Discrete):
            top_action = np.argmax(action_values)
            if np.random.rand() < epsilon:
                return self.action_space.sample()
            else:
                return top_action
        else:
            noise = np.random.randn(1, self.action_space.shape) * self.variance_schedule.current_value * \
                    self.action_space.max_abs_range
            return np.squeeze(action_values + (np.random.rand() < epsilon) * noise)

    def get_control_param(self):
        return self.evaluation_epsilon if self.phase == RunPhase.TEST else self.epsilon_schedule.current_value
