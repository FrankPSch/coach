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
from typing import Union

from agents.agent import Agent
from utils import Enum
from logger import screen
from collections import OrderedDict
import numpy as np


class PolicyGradientRescaler(Enum):
    TOTAL_RETURN = 0
    FUTURE_RETURN = 1
    FUTURE_RETURN_NORMALIZED_BY_EPISODE = 2
    FUTURE_RETURN_NORMALIZED_BY_TIMESTEP = 3  # baselined
    Q_VALUE = 4
    A_VALUE = 5
    TD_RESIDUAL = 6
    DISCOUNTED_TD_RESIDUAL = 7
    GAE = 8
    CUSTOM_ACTOR_CRITIC = 9


class PolicyOptimizationAgent(Agent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)

        self.policy_gradient_rescaler = None
        if hasattr(self.ap.algorithm, 'policy_gradient_rescaler'):
            self.policy_gradient_rescaler = self.ap.algorithm.policy_gradient_rescaler

        # statistics for variance reduction
        self.last_gradient_update_step_idx = 0
        self.max_episode_length = 100000
        self.mean_return_over_multiple_episodes = np.zeros(self.max_episode_length)
        self.num_episodes_where_step_has_been_seen = np.zeros(self.max_episode_length)
        self.entropy = self.register_signal('Entropy')

    def log_to_screen(self):
        # log to screen
        log = OrderedDict()
        if self.task_id is not None:
            log["Worker"] = self.task_id
        log["Episode"] = self.current_episode
        log["Total reward"] = round(self.total_reward_in_current_episode, 2)
        log["Steps"] = self.total_steps_counter
        log["Training iteration"] = self.training_iteration
        screen.log_dict(log, prefix=self.phase.value)

    def update_episode_statistics(self, episode):
        episode_discounted_returns = []
        for i in range(episode.length()):
            transition = episode.get_transition(i)
            episode_discounted_returns.append(transition.total_return)
            self.num_episodes_where_step_has_been_seen[i] += 1
            self.mean_return_over_multiple_episodes[i] -= self.mean_return_over_multiple_episodes[i] / \
                                                          self.num_episodes_where_step_has_been_seen[i]
            self.mean_return_over_multiple_episodes[i] += transition.total_return / \
                                                          self.num_episodes_where_step_has_been_seen[i]
        self.mean_discounted_return = np.mean(episode_discounted_returns)
        self.std_discounted_return = np.std(episode_discounted_returns)

    def train(self):
        episode = self.memory.get_episode(0)

        if not episode:
            return

        # check if we should calculate gradients or skip
        episode_ended = self.memory.num_complete_episodes() >= 1
        num_steps_passed_since_last_update = episode.length() - self.last_gradient_update_step_idx
        is_t_max_steps_passed = num_steps_passed_since_last_update >= self.ap.algorithm.num_steps_between_gradient_updates
        if not (is_t_max_steps_passed or episode_ended):
            return 0

        total_loss = 0
        if num_steps_passed_since_last_update > 0:

            # we need to update the returns of the episode until now
            episode.update_returns(self.ap.algorithm.discount)

            # get t_max transitions or less if the we got to a terminal state
            # will be used for both actor-critic and vanilla PG.
            # # In order to get full episodes, Vanilla PG will set the end_idx to a very big value.
            transitions = []
            start_idx = self.last_gradient_update_step_idx
            end_idx = episode.length()

            for idx in range(start_idx, end_idx):
                transitions.append(episode.get_transition(idx))
            self.last_gradient_update_step_idx = end_idx

            # update the statistics for the variance reduction techniques
            if self.policy_gradient_rescaler in \
                    [PolicyGradientRescaler.FUTURE_RETURN_NORMALIZED_BY_EPISODE,
                     PolicyGradientRescaler.FUTURE_RETURN_NORMALIZED_BY_TIMESTEP]:
                self.update_episode_statistics(episode)

            # accumulate the gradients and apply them once in every apply_gradients_every_x_episodes episodes
            total_loss, losses, unclipped_grads = self.learn_from_batch(transitions)
            if self.current_episode % self.ap.algorithm.apply_gradients_every_x_episodes == 0:
                for network in self.networks.values():
                    network.apply_gradients_and_sync_networks()
            self.training_iteration += 1

        # move the pointer to the next episode start and discard the episode.
        if episode_ended:
            # we need to remove the episode, because the next training iteration will be called before storing any
            # additional transitions in the memory (we don't store a transition for the first call to observe), so the
            # length of the memory won't be enforced and the old episode won't be removed
            self.memory.remove_episode(0)
            self.last_gradient_update_step_idx = 0

        return total_loss
