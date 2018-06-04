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

from memories.memory import Memory, Episode, MemoryGranularity, MemoryParameters
from core_types import Transition
from typing import List, Tuple, Union, Dict, Any


class EpisodicExperienceReplayParameters(MemoryParameters):
    def __init__(self):
        super().__init__()
        self.max_size = (MemoryGranularity.Transitions, 1000000)
        self.discount = 0.99
        self.bootstrap_total_return_from_old_policy = False
        self.n_step = -1
        self.num_predicted_steps_ahead = None

    @property
    def path(self):
        return 'memories.episodic_experience_replay:EpisodicExperienceReplay'


class EpisodicExperienceReplay(Memory):
    """
    A replay buffer that stores episodes of transitions. The additional structure allows performing various
    calculations of total return and other values that depend on the sequential behavior of the transitions
    in the episode.
    """
    def __init__(self, max_size: Tuple[MemoryGranularity, int], discount: float,
                 bootstrap_total_return_from_old_policy: bool, n_step: int,
                 num_predicted_steps_ahead: Union[None, int]):
        """
        :param max_size: the maximum number of transitions or episodes to hold in the memory
        :param discount: the discount factor to use when calculating total returns
        :param bootstrap_total_return_from_old_policy: should the total return be bootstrapped from the values in the
                                                       memory
        :param n_step: the number of future steps to sum the reward over before bootstrapping
        :param num_predicted_steps_ahead: for each transition, attach the future measurements for
                                          num_predicted_steps_ahead future steps (used by DFP)

        """
        super().__init__(max_size)
        self.discount = discount
        self.n_step = n_step
        self.num_predicted_steps_ahead = num_predicted_steps_ahead

        self._buffer = [Episode()]  # list of episodes
        self.transitions = []
        self._length = 1  # the episodic replay buffer starts with a single empty episode
        self._num_transitions = 0
        self._num_transitions_in_complete_episodes = 0
        self.return_is_bootstrapped = bootstrap_total_return_from_old_policy

    def length(self) -> int:
        """
        Get the number of episodes in the ER (even if they are not complete)
        """
        if self._length is not 0 and self._buffer[-1].is_empty():
            return self._length - 1
        return self._length

    def num_complete_episodes(self):
        """ Get the number of complete episodes in ER """
        return self._length - 1

    def num_transitions(self):
        return self._num_transitions

    def num_transitions_in_complete_episodes(self):
        return self._num_transitions_in_complete_episodes

    def sample(self, size: int) -> List[Transition]:
        """
        Sample a batch of transitions form the replay buffer. If the requested size is larger than the number
        of samples available in the replay buffer then the batch will return empty.
        :param size: the size of the batch to sample
        :return: a batch (list) of selected transitions from the replay buffer
        """
        if self.num_complete_episodes() >= 1:
            transitions_idx = np.random.randint(self.num_transitions_in_complete_episodes(), size=size)
            batch = [self.transitions[i] for i in transitions_idx]
            return batch
        else:
            raise ValueError("The episodic replay buffer cannot be sampled since there are no complete episodes yet. "
                             "There is currently 1 episodes with {} transitions".format(self._buffer[0].length()))

    def enforce_max_length(self) -> None:
        """
        Make sure that the size of the replay buffer does not pass the maximum size allowed.
        If it passes the max size, the oldest episode in the replay buffer will be removed.
        :return: None
        """
        granularity, size = self.max_size
        if granularity == MemoryGranularity.Transitions:
            while size != 0 and self.num_transitions() > size:
                self.remove_episode(0)
        elif granularity == MemoryGranularity.Episodes:
            while self.length() > size:
                self.remove_episode(0)

    def _update_episode(self, episode: Episode) -> None:
        episode.update_returns(self.discount, is_bootstrapped=self.return_is_bootstrapped,
                               n_step_return=self.n_step)
        if self.num_predicted_steps_ahead is not None:
            episode.update_measurements_targets(self.num_predicted_steps_ahead)

    def verify_last_episode_is_closed(self) -> None:
        """
        Verify that there is no open episodes in the replay buffer
        :return: None
        """
        last_episode = self.get(-1)
        if last_episode and last_episode.length() > 0:
            self.close_last_episode()

    def close_last_episode(self) -> None:
        """
        Close the last episode in the replay buffer and open a new one
        :return: None
        """
        last_episode = self._buffer[-1]

        self._num_transitions_in_complete_episodes += last_episode.length()
        self._length += 1

        # create a new Episode for the next transitions to be placed into
        self._buffer.append(Episode())

        # if update episode adds to the buffer, a new Episode needs to be ready first
        # it would be better if this were less state full
        self._update_episode(last_episode)

        self.enforce_max_length()

    def store(self, transition: Transition) -> None:
        """
        Store a new transition in the memory. If the transition game_over flag is on, this closes the episode and
        creates a new empty episode.
        :param transition: a transition to store
        :return: None
        """
        if len(self._buffer) == 0:
            self._buffer.append(Episode())
        last_episode = self._buffer[-1]
        last_episode.insert(transition)
        self.transitions.append(transition)
        self._num_transitions += 1
        if transition.game_over:
            self.close_last_episode()

        self.enforce_max_length()

    def get_episode(self, episode_index: int) -> Union[None, Episode]:
        """
        Returns the episode in the given index. If the episode does not exist, returns None instead.
        :param episode_index: the index of the episode to return
        :return: the corresponding episode
        """
        if self.length() == 0 or episode_index >= self.length():
            return None
        episode = self._buffer[episode_index]
        return episode

    def remove_episode(self, episode_index: int) -> None:
        """
        Remove the episode in the given index (even if it is not complete yet)
        :param episode_index: the index of the episode to remove
        :return: None
        """
        if len(self._buffer) > episode_index:
            episode_length = self._buffer[episode_index].length()
            self._length -= 1
            self._num_transitions -= episode_length
            self._num_transitions_in_complete_episodes -= episode_length
            del self.transitions[:episode_length]
            del self._buffer[episode_index]

    # for API compatibility
    def get(self, episode_index: int) -> Union[None, Episode]:
        """
        Returns the episode in the given index. If the episode does not exist, returns None instead.
        :param episode_index: the index of the episode to return
        :return: the corresponding episode
        """
        return self.get_episode(episode_index)

    def get_last_complete_episode(self) -> Union[None, Episode]:
        """
        Returns the last complete episode in the memory or None if there are no complete episodes
        :return: None or the last complete episode
        """
        last_complete_episode_index = self.num_complete_episodes() - 1
        if last_complete_episode_index >= 0:
            return self.get(last_complete_episode_index)
        else:
            return None

    # for API compatibility
    def remove(self, episode_index: int):
        """
        Remove the episode in the given index (even if it is not complete yet)
        :param episode_index: the index of the episode to remove
        :return: None
        """
        self.remove_episode(episode_index)

    def update_last_transition_info(self, info: Dict[str, Any]) -> None:
        """
        Update the info of the last transition stored in the memory
        :param info: the new info to append to the existing info
        :return: None
        """
        episode = self._buffer[-1]
        if episode.length() == 0:
            if len(self._buffer) < 2:
                return
            episode = self._buffer[-2]
        episode.transitions[-1].info.update(info)

    def clean(self) -> None:
        """
        Clean the memory by removing all the episodes
        :return: None
        """
        self.transitions = []
        self._buffer = [Episode()]
        self._length = 1
        self._num_transitions = 0
        self._num_transitions_in_complete_episodes = 0

    def mean_reward(self) -> np.ndarray:
        """
        Get the mean reward in the replay buffer
        :return: the mean reward
        """
        return np.mean([transition.reward for transition in self.transitions])
