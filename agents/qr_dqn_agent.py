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

from agents.dqn_agent import DQNAgentParameters, DQNNetworkParameters, DQNAlgorithmParameters
from configurations import OutputTypes
from core_types import StateType
from schedules import LinearSchedule


class QuantileRegressionDQNNetworkParameters(DQNNetworkParameters):
    def __init__(self):
        super().__init__()
        self.output_types = [OutputTypes.QuantileRegressionQ]
        self.learning_rate = 0.00005
        self.optimizer_epsilon = 0.01 / 32
        self.neon_support = False


class QuantileRegressionDQNAlgorithmParameters(DQNAlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.atoms = 200
        self.huber_loss_interval = 1  # called k in the paper


class QuantileRegressionDQNAgentParameters(DQNAgentParameters):
    def __init__(self):
        super().__init__()
        self.algorithm = QuantileRegressionDQNAlgorithmParameters()
        self.network_wrappers = {"main": QuantileRegressionDQNNetworkParameters()}
        self.exploration.epsilon_schedule = LinearSchedule(1, 0.01, 1000000)
        self.exploration.evaluation_epsilon = 0.001

    @property
    def path(self):
        return 'agents.qr_dqn_agent:QuantileRegressionDQNAgent'


# Quantile Regression Deep Q Network - https://arxiv.org/pdf/1710.10044v1.pdf
class QuantileRegressionDQNAgent(ValueOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.quantile_probabilities = np.ones(self.ap.algorithm.atoms) / float(self.ap.algorithm.atoms)

    def get_q_values(self, quantile_values):
        return np.dot(quantile_values, self.quantile_probabilities)

    # prediction's format is (batch,actions,atoms)
    def get_all_q_values_for_states(self, states: StateType):
        quantile_values = self.get_prediction(states)
        return self.get_q_values(quantile_values)

    def learn_from_batch(self, batch):
        current_states, next_states, actions, rewards, game_overs, _ = self.extract_batch(batch, 'main')

        # get the quantiles of the next states and current states
        next_state_quantiles = self.networks['main'].target_network.predict(next_states)
        current_quantiles = self.networks['main'].online_network.predict(current_states)

        # get the optimal actions to take for the next states
        target_actions = np.argmax(self.get_q_values(next_state_quantiles), axis=1)

        # calculate the Bellman update
        batch_idx = list(range(self.ap.network_wrappers['main'].batch_size))
        rewards = np.expand_dims(rewards, -1)
        game_overs = np.expand_dims(game_overs, -1)
        TD_targets = rewards + (1.0 - game_overs) * self.ap.algorithm.discount \
                               * next_state_quantiles[batch_idx, target_actions]

        # get the locations of the selected actions within the batch for indexing purposes
        actions_locations = [[b, a] for b, a in zip(batch_idx, actions)]

        # calculate the cumulative quantile probabilities and reorder them to fit the sorted quantiles order
        cumulative_probabilities = np.array(range(self.ap.algorithm.atoms + 1)) / float(self.ap.algorithm.atoms) # tau_i
        quantile_midpoints = 0.5*(cumulative_probabilities[1:] + cumulative_probabilities[:-1])  # tau^hat_i
        quantile_midpoints = np.tile(quantile_midpoints, (self.ap.network_wrappers['main'].batch_size, 1))
        sorted_quantiles = np.argsort(current_quantiles[batch_idx, actions])
        for idx in range(self.ap.network_wrappers['main'].batch_size):
            quantile_midpoints[idx, :] = quantile_midpoints[idx, sorted_quantiles[idx]]

        # train
        result = self.networks['main'].train_and_sync_networks({
            **current_states,
            'output_0_0': actions_locations,
            'output_0_1': quantile_midpoints,
        }, TD_targets)
        total_loss, losses, unclipped_grads = result[:3]

        return total_loss, losses, unclipped_grads

