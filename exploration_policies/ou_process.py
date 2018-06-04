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
from exploration_policies.exploration_policy import ExplorationPolicy, ExplorationParameters
from spaces import ActionSpace
from core_types import RunPhase, ActionType
from typing import List


# Based on on the description in:
# https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OUProcessParameters(ExplorationParameters):
    def __init__(self):
        super().__init__()
        self.mu = 0
        self.theta = 0.15
        self.sigma = 0.2
        self.dt = 0.01

    @property
    def path(self):
        return 'exploration_policies.ou_process:OUProcess'


# Ornstein-Uhlenbeck process
class OUProcess(ExplorationPolicy):
    def __init__(self, action_space: ActionSpace, mu: float, theta: float, sigma: float, dt: float):
        """
        :param action_space: the action space used by the environment
        """
        super().__init__(action_space)
        self.mu = float(mu) * np.ones(self.action_space.shape)
        self.theta = float(theta)
        self.sigma = float(sigma) * np.ones(self.action_space.shape)
        self.state = np.zeros(self.action_space.shape)
        self.dt = dt

    def reset(self):
        self.state = np.zeros(self.action_space.shape)

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state

    def get_action(self, action_values: List[ActionType]) -> ActionType:
        if self.phase == RunPhase.TRAIN:
            noise = self.noise()
        else:
            noise = np.zeros(self.action_space.shape)

        action = action_values.squeeze() + noise
        action = self.action_space.clip_action_to_space(action)
        return action

    def get_control_param(self):
        if self.phase == RunPhase.TRAIN:
            return self.state
        else:
            return np.zeros(self.action_space.shape)
