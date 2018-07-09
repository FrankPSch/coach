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
from enum import Enum
from typing import Tuple, List

import numpy as np

from core_types import GoalTypes, Transition
from memories.experience_replay import ExperienceReplayParameters, ExperienceReplay, MemoryGranularity
from spaces import GoalsActionSpace, ReachingGoal


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
        self.goals_action_space = None
        self.goal_replay = True
        self.action_replay = False

    @property
    def path(self):
        return 'memories.hindsight_experience_replay:HindsightExperienceReplay'


class HindsightExperienceReplay(ExperienceReplay):
    """
    Implements Hindsight Experience Replay as described in the following paper: https://arxiv.org/pdf/1707.01495.pdf

    Warning! will not work correctly with multiple workers if the memory is shared between them, since the episodes
    transitions get messed up. To work with multiple workers, please use the EpisodicHindsightExperienceReplay memory
    instead.
    """
    def __init__(self, max_size: Tuple[MemoryGranularity, int],
                 hindsight_transitions_per_regular_transition: int,
                 hindsight_goal_selection_method: HindsightGoalSelectionMethod,
                 goals_action_space: GoalsActionSpace,
                 goal_replay: bool=True,
                 action_replay: bool=False,
                 allow_duplicates_in_batch_sampling: bool=True):
        """
        :param max_size: The maximum size of the memory. should be defined in a granularity of Transitions
        :param hindsight_transitions_per_regular_transition: The number of hindsight artificial transitions to generate
                                                             for each actual transition
        :param hindsight_goal_selection_method: The method that will be used for generating the goals for the
                                                hindsight transitions. Should be one of HindsightGoalSelectionMethod
        :param goals_action_space: A GoalsActionSpace  which defines the properties of the goals
        :param goal_replay: Generate hindsight transitions where the goal is replaced by the actual goal achieved
                            (as defined in the original HER paper)
        :param action_replay: Replace the actions in the transitions with the actual next states achieved
                              (as defined in the 'Hierarchical Reinforcement Learning with Hindsight' paper)
                              https://arxiv.org/abs/1805.08180
                              ### GAL: this is confusing, maybe better document it? which exact paper is 'HRL with Hindsight paper'?
        """
        super().__init__(max_size, allow_duplicates_in_batch_sampling)

        self.hindsight_transitions_per_regular_transition = hindsight_transitions_per_regular_transition
        self.hindsight_goal_selection_method = hindsight_goal_selection_method
        self.goals_action_space = goals_action_space
        self.goal_replay = goal_replay
        self.action_replay = action_replay

        self.last_episode_start_idx = 0

    def _sample_goal(self, episode_transitions: List, transition_index: int):
        """
        Sample a single goal state according to the sampling method
        :param episode_transitions: a list of all the transitions in the current episode
        :param transition_index: the transition to start sampling from
        :return: a goal corresponding to the sampled state
        """
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
        return self.goals_action_space.goal_from_state(selected_transition.state)

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
        self.reader_writer_lock.lock_writing_and_reading()

        super().store(transition, lock=False)

        # used primarily in HRL, where the action of the high level policy is the goal of the low level policy.
        # we then replace the action of the high level with the actual goal that was reached, which is the actual next
        # state
        if self.action_replay:
            self.transitions[-1].action = transition.next_state[self.goals_action_space.goal_type]

        # generate hindsight transitions only when an episode is finished
        if transition.game_over:
            if self.goal_replay:
                # generate hindsight experience replay goals
                last_episode_transitions = self.transitions[self.last_episode_start_idx:]

                # cannot create a future hindsight goal in the last transition of an episode
                if self.hindsight_goal_selection_method == HindsightGoalSelectionMethod.Future:
                    relevant_base_transitions = self.transitions[self.last_episode_start_idx:-1]
                else:
                    relevant_base_transitions = self.transitions[self.last_episode_start_idx:]

                # for each transition in the last episode, create a set of hindsight transitions
                for transition_index, transition in enumerate(relevant_base_transitions):
                    sampled_goals = self._sample_goals(last_episode_transitions, transition_index)
                    for goal in sampled_goals:
                        hindsight_transition = copy.copy(transition)

                        if hindsight_transition.state['desired_goal'].shape != goal.shape:
                            raise ValueError((
                                'goal shape {goal_shape} already in transition is '
                                'different than the one sampled as a hindsight goal '
                                '{hindsight_goal_shape}.'
                            ).format(
                                goal_shape=hindsight_transition.state['desired_goal'].shape,
                                hindsight_goal_shape=goal.shape,
                            ))

                        # update the goal in the transition
                        ### TODO - Gal : didn't we want to have the goal only as part of the transition and not part of the state? why have it 3 times?
                        hindsight_transition.goal = goal
                        hindsight_transition.state['desired_goal'] = goal
                        hindsight_transition.next_state['desired_goal'] = goal

                        # update the reward and terminal signal according to the goal
                        hindsight_transition.reward, hindsight_transition.game_over = \
                            self.goals_action_space.get_reward_for_goal_and_state(goal, hindsight_transition.next_state)

                        hindsight_transition.total_return = None
                        self._num_transitions += 1
                        self.transitions.append(hindsight_transition)

            self._enforce_max_length()

            self.last_episode_start_idx = self.num_transitions()

        self.reader_writer_lock.release_writing_and_reading()
