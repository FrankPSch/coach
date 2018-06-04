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

from agents.value_optimization_agent import ValueOptimizationAgent
import numpy as np
from agents.dqn_agent import DQNAgentParameters, DQNAlgorithmParameters


class MixedMonteCarloAlgorithmParameters(DQNAlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.monte_carlo_mixing_rate = 0.1


class MixedMonteCarloAgentParameters(DQNAgentParameters):
    def __init__(self):
        super().__init__()
        self.algorithm = MixedMonteCarloAlgorithmParameters()

    @property
    def path(self):
        return 'agents.mmc_agent:MixedMonteCarloAgent'


class MixedMonteCarloAgent(ValueOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.mixing_rate = agent_parameters.algorithm.monte_carlo_mixing_rate

    def learn_from_batch(self, batch):
        current_states, next_states, actions, rewards, game_overs, total_return = self.extract_batch(batch, 'main')

        TD_targets = self.networks['main'].online_network.predict(current_states)
        selected_actions = np.argmax(self.networks['main'].online_network.predict(next_states), 1)
        q_st_plus_1 = self.networks['main'].target_network.predict(next_states)
        # initialize with the current prediction so that we will
        #  only update the action that we have actually done in this transition
        for i in range(self.ap.network_wrappers['main'].batch_size):
            one_step_target = rewards[i] + (1.0 - game_overs[i]) * self.ap.algorithm.discount * q_st_plus_1[i][
                selected_actions[i]]
            monte_carlo_target = total_return[i]
            TD_targets[i, actions[i]] = (1 - self.mixing_rate) * one_step_target + self.mixing_rate * monte_carlo_target

        result = self.networks['main'].train_and_sync_networks(current_states, TD_targets)
        total_loss, losses, unclipped_grads = result[:3]

        return total_loss, losses, unclipped_grads
