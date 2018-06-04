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
from typing import List, Dict, Union

import numpy as np
from agents.agent import Agent
from core_types import ActionInfo, StateType
from memories.prioritized_experience_replay import PrioritizedExperienceReplay
from spaces import Discrete


class ValueOptimizationAgent(Agent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.q_values = self.register_signal("Q")
        self.q_value_for_action = {}

    def init_environment_dependent_modules(self):
        super().init_environment_dependent_modules()
        if isinstance(self.spaces.action, Discrete):
            for i in range(len(self.spaces.action.actions)):
                self.q_value_for_action[i] = self.register_signal("Q for action {}".format(i),
                                                                  dump_one_value_per_episode=False,
                                                                  dump_one_value_per_step=True)

    # Algorithms for which q_values are calculated from predictions will override this function
    def get_all_q_values_for_states(self, states: StateType):
        return self.get_prediction(states)

    def get_prediction(self, states):
        return self.networks['main'].online_network.predict(self.dict_state_to_batches_dict(states, 'main'))

    def update_transition_priorities_and_get_weights(self, TD_errors, batch):
        # update errors in prioritized replay buffer
        importance_weights = None
        if isinstance(self.memory, PrioritizedExperienceReplay):
            self.memory.update_priorities([t.info['idx'] for t in batch], TD_errors)
            importance_weights = np.array([t.info['weight'] for t in batch])
        return importance_weights

    def _validate_action(self, policy, action):
        if np.array(action).shape != ():
            raise ValueError((
                'The exploration_policy {} returned a vector of actions '
                'instead of a single action. ValueOptimizationAgents '
                'require exploration policies which return a single action.'
            ).format(policy.__class__.__name__))

    def choose_action(self, curr_state):
        actions_q_values = self.get_all_q_values_for_states(curr_state)

        # choose action according to the exploration policy and the current phase (evaluating or training the agent)
        action = self.exploration_policy.get_action(actions_q_values)
        self._validate_action(self.exploration_policy, action)

        # this is for bootstrapped dqn
        if type(actions_q_values) == list and len(actions_q_values) > 0:
            actions_q_values = self.exploration_policy.last_action_values
        actions_q_values = actions_q_values.squeeze()

        # store the q values statistics for logging
        self.q_values.add_sample(actions_q_values)
        for i, q_value in enumerate(actions_q_values):
            self.q_value_for_action[i].add_sample(q_value)

        action_info = ActionInfo(action=action,
                                 action_value=actions_q_values[action],
                                 max_action_value=np.max(actions_q_values))
        return action_info
