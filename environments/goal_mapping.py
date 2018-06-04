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
import scipy
from scipy import spatial

from core_types import GoalTypes


class GoalMapping(object):
    def __init__(self, goal_type: str, distance_from_goal_threshold=0.1, agent=None):
        """
        :param goal_type: the state value to use for checking if the goal was reached.
                                   can be embedding, observation or measurements
        :param agent: when a state embedding needs to be used, the embedding in this agent
                      will be used
        """

        self._goal_type = goal_type
        self._distance_from_goal_threshold = distance_from_goal_threshold
        self._agent = agent

    def goal_from_state(self, state):
        """
        Given a state, map it into goal space
        """
        return state[self._goal_type]
        # if self._goal_type == GoalTypes.Embedding:
        #     return self._agent.get_state_embedding(state)
        # # TODO: instead of converting from relative to absolute here, convert the goal
        # # from relative to absolute when it is created
        # # elif self._goal_type == GoalTypes.EmbeddingChange:
        # #     return self._agent.get_state_embedding(state) - self.initial_state_embedding
        # elif self._goal_type == GoalTypes.Observation:
        #     # only use the most recent observation, not the entire frame stack
        #     return state['observation'][..., -1]
        # elif self._goal_type == GoalTypes.Measurements:
        #     return state['measurements']
        # else:
        #     raise ValueError("The given state value ({}) is not a valid value".format(self._goal_type))

    def goal_reached(self, goal: np.ndarray, state: dict) -> bool:
        """
        Given an embedding, check if the goal was reached
        :param goal: a numpy array representing the goal
        :param state: a dict representing the state
        :return: a boolean indicating if the goal was reached
        """
        state_value = self.goal_from_state(state)

        # calculate distance
        if self._goal_type == GoalTypes.EmbeddingChange:
            dist = scipy.spatial.distance.cosine(goal, state_value)
        else:
            dist = scipy.spatial.distance.euclidean(goal, state_value)

        return dist <= self._distance_from_goal_threshold

    def set_agent(self, agent):
        """
        :param agent: when a state embedding needs to be used, the embedding in this agent
                      will be used
        """
        if self._agent is not None:
            raise ValueError('agent should probably not be changed once it has already been set')

        self._agent = agent