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

import numpy as np
from scipy.stats import truncnorm

from core_types import RunPhase, ActionType
from exploration_policies.exploration_policy import ExplorationPolicy, ExplorationParameters
from schedules import Schedule, LinearSchedule
from spaces import ActionSpace, BoxActionSpace


class TruncatedNormalParameters(ExplorationParameters):
    def __init__(self):
        super().__init__()
        self.noise_percentage_schedule = LinearSchedule(0.1, 0.1, 50000)
        self.evaluation_noise_percentage = 0.05
        self.clip_low = 0
        self.clip_high = 1

    @property
    def path(self):
        return 'exploration_policies.truncated_normal:TruncatedNormal'


class TruncatedNormal(ExplorationPolicy):
    def __init__(self, action_space: ActionSpace, noise_percentage_schedule: Schedule,
                 evaluation_noise_percentage: float, clip_low: float, clip_high: float):
        """
        :param action_space: the action space used by the environment
        :param noise_percentage_schedule: the schedule for the noise variance percentage relative to the absolute range
                                          of the action space
        :param evaluation_noise_percentage: the noise variance percentage that will be used during evaluation phases
        """
        super().__init__(action_space)
        self.noise_percentage_schedule = noise_percentage_schedule
        self.evaluation_noise_percentage = evaluation_noise_percentage
        self.clip_low = clip_low
        self.clip_high = clip_high

        if not isinstance(action_space, BoxActionSpace):
            raise ValueError("Truncated normal exploration works only for continuous controls."
                             "The given action space is of type: {}".format(action_space.__class__.__name__))

        if not np.all(-np.inf < action_space.high) or not np.all(action_space.high < np.inf)\
                or not np.all(-np.inf < action_space.low) or not np.all(action_space.low < np.inf):
            raise ValueError("Additive noise exploration requires bounded actions")

        # TODO: allow working with unbounded actions by defining the noise in terms of range and not percentage

    def get_action(self, action_values: List[ActionType]) -> ActionType:
        # set the current noise percentage
        if self.phase == RunPhase.TEST:
            current_noise_precentage = self.evaluation_noise_percentage
        else:
            current_noise_precentage = self.noise_percentage_schedule.current_value

        # scale the noise to the action space range
        action_values_std = 2 * current_noise_precentage * self.action_space.max_abs_range

        # extract the mean values
        if isinstance(action_values, list):
            # the action values are expected to be a list with the action mean and optionally the action stdev
            action_values_mean = action_values[0].squeeze()
        else:
            # the action values are expected to be a numpy array representing the action mean
            action_values_mean = action_values.squeeze()

        # step the noise schedule
        if self.phase == RunPhase.TRAIN:
            self.noise_percentage_schedule.step()
            # the second element of the list is assumed to be the standard deviation
            if isinstance(action_values, list) and len(action_values) > 1:
                action_values_std = action_values[1].squeeze()

        # sample from truncated normal distribution
        distribution = truncnorm(self.clip_low, self.clip_high, loc=action_values_mean, scale=action_values_std)
        action = distribution.rvs(1)

        action = self.action_space.clip_action_to_space(action)

        return action

    def get_control_param(self):
        return np.ones(self.action_space.shape)*self.noise_percentage_schedule.current_value
