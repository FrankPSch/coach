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

import copy
import numpy as np

from memories.experience_replay import ExperienceReplayParameters, ExperienceReplay, MemoryGranularity
from environments.goal_mapping import GoalMapping
from core_types import GoalTypes, Transition
from typing import Tuple, List
from enum import Enum


class HindsightGoalSelectionMethod(Enum):
    Future = 0
    Final = 1
    Episode = 2
    Random = 3


class HindsightExperienceReplayParameters(ExperienceReplayParameters):
    def __init__(self):
        super().__init__()
        self.hindsight_transitions_per_regular_transition = None
        self.hindsight_goal_selection_method = None
        self.goal_type = None
        self.distance_from_goal_threshold = None

    @property
    def path(self):
        return 'memories.hindsight_experience_replay:HindsightExperienceReplay'


class HindsightExperienceReplay(ExperienceReplay):
    def __init__(self, max_size: Tuple[MemoryGranularity, int],
                 hindsight_transitions_per_regular_transition: int,
                 hindsight_goal_selection_method: HindsightGoalSelectionMethod,
                 goal_type: GoalTypes,
                 distance_from_goal_threshold: float):
        super().__init__(max_size)

        self.hindsight_transitions_per_regular_transition = hindsight_transitions_per_regular_transition
        self.hindsight_goal_selection_method = hindsight_goal_selection_method

        # TODO: keep consistent with agent
        # TODO: agent object is not available here, this should probably be created somewhere else and passed in
        self.goal_mapping = GoalMapping(
            goal_type=goal_type,
            distance_from_goal_threshold=distance_from_goal_threshold,
        )

        self.last_episode_start_idx = 0

    def _sample_goal(self, episode_transitions: List, transition_index: int):
        """
        Sample a single goal state according to the sampling method
        :param episode_transitions: a list of all the transitions in the current episode
        :param transition_index: the transition to start sampling from
        :return: a goal corresponding to the sampled state
        """
        selected_transition = None
        if self.hindsight_goal_selection_method == HindsightGoalSelectionMethod.Future:
            # states that were observed in the same episode after the transition that is being replayed
            selected_transition = np.random.choice(episode_transitions[transition_index+1:])
        elif self.hindsight_goal_selection_method == HindsightGoalSelectionMethod.Final:
            # the final state in the episode
            selected_transition = episode_transitions[-1]
        elif self.hindsight_goal_selection_method == HindsightGoalSelectionMethod.Episode:
            # a random state from the episode
            selected_transition = np.random.choice(episode_transitions)
        elif self.hindsight_goal_selection_method == HindsightGoalSelectionMethod.Random:
            # a random state from the entire replay buffer
            selected_transition = np.random.choice(self.transitions)
        else:
            raise ValueError("Invalid goal selection method was used for the hindsight goal selection")
        return self.goal_mapping.goal_from_state(selected_transition.state)

    def _sample_goals(self, episode_transitions: List, transition_index: int):
        """
        Sample a batch of goal states according to the sampling method
        :param episode_transitions: a list of all the transitions in the current episode
        :param transition_index: the transition to start sampling from
        :return: a goal corresponding to the sampled state
        """
        return [
            self._sample_goal(episode_transitions, transition_index)
            for _ in range(self.hindsight_transitions_per_regular_transition)
        ]

    def store(self, transition: Transition) -> None:
        super().store(transition)

        # generate hindsight transitions only when an episode is finished
        if self.get(-1).game_over:
            # generate hindsight experience replay goals
            # cannot create a future hindsight goal in the last transition of an episode
            last_episode_transitions = self.transitions[self.last_episode_start_idx:]
            for transition_index, transition in enumerate(last_episode_transitions[:-1]):
                sampled_goals = self._sample_goals(last_episode_transitions, transition_index)
                for goal in sampled_goals:
                    hindsight_transition = copy.deepcopy(transition)

                    if hindsight_transition.state['goal'].shape != goal.shape:
                        raise ValueError((
                            'goal shape {goal_shape} already in transition is '
                            'different than the one sampled as a hindsight goal '
                            '{hindsight_goal_shape}.'
                        ).format(
                            goal_shape=hindsight_transition.state['goal'].shape,
                            hindsight_goal_shape=goal.shape,
                        ))

                    # update the goal in the transition
                    hindsight_transition.goal = goal
                    hindsight_transition.state['goal'] = goal
                    hindsight_transition.next_state['goal'] = goal

                    # update the reward and terminal signal according to the goal
                    if self.goal_mapping.goal_reached(goal, hindsight_transition.next_state):
                        hindsight_transition.reward = 0
                        hindsight_transition.game_over = True
                    else:
                        hindsight_transition.reward = -1
                        hindsight_transition.game_over = False

                    hindsight_transition.total_return = None
                    self._num_transitions += 1
                    self.transitions.append(hindsight_transition)

            self.enforce_max_length()

            self.last_episode_start_idx = self.num_transitions()
