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
from configurations import AlgorithmParameters, NetworkParameters, InputTypes, OutputTypes, MiddlewareTypes, \
    AgentParameters, InputEmbedderParameters
from configurations import EmbedderScheme
from core_types import EnvironmentSteps
from exploration_policies.e_greedy import EGreedyParameters
from memories.episodic_experience_replay import EpisodicExperienceReplayParameters
from memories.experience_replay import ExperienceReplayParameters
from memories.prioritized_experience_replay import PrioritizedExperienceReplay
import numpy as np

from schedules import LinearSchedule


class DQNAlgorithmParameters(AlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(10000)
        self.num_consecutive_playing_steps = EnvironmentSteps(4)
        self.discount = 0.99


class DQNNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_types = {'observation': InputEmbedderParameters()}
        self.middleware_type = MiddlewareTypes.FC
        self.embedder_scheme = EmbedderScheme.Medium
        self.output_types = [OutputTypes.Q]
        self.loss_weights = [1.0]
        self.optimizer_type = 'Adam'
        self.batch_size = 32
        self.hidden_layers_activation_function = 'relu'
        self.replace_mse_with_huber_loss = True
        self.neon_support = True
        self.async_training = True
        self.shared_optimizer = True
        self.create_target_network = True


class DQNAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=DQNAlgorithmParameters(),
                         exploration=EGreedyParameters(),
                         memory=ExperienceReplayParameters(),
                         networks={"main": DQNNetworkParameters()})
        self.exploration.epsilon_schedule = LinearSchedule(1, 0.1, 1000000)
        self.exploration.evaluation_epsilon = 0.05

    @property
    def path(self):
        return 'agents.dqn_agent:DQNAgent'


# Deep Q Network - https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
class DQNAgent(ValueOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)

    def learn_from_batch(self, batch):
        current_states, next_states, actions, rewards, game_overs, _ = self.extract_batch(batch, 'main')

        # for the action we actually took, the error is:
        # TD error = r + discount*max(q_st_plus_1) - q_st
        # for all other actions, the error is 0
        q_st_plus_1 = self.networks['main'].target_network.predict(next_states)
        # initialize with the current prediction so that we will
        TD_targets = self.networks['main'].online_network.predict(current_states)

        #  only update the action that we have actually done in this transition
        TD_errors = []
        for i in range(self.ap.network_wrappers['main'].batch_size):
            new_target = rewards[i] + (1.0 - game_overs[i]) * self.ap.algorithm.discount * np.max(q_st_plus_1[i], 0)
            TD_errors.append(np.abs(new_target - TD_targets[i, actions[i]]))
            TD_targets[i, actions[i]] = new_target

        # update errors in prioritized replay buffer
        importance_weights = self.update_transition_priorities_and_get_weights(TD_errors, batch)

        result = self.networks['main'].train_and_sync_networks(current_states, TD_targets,
                                                               importance_weights=importance_weights)
        total_loss, losses, unclipped_grads = result[:3]

        return total_loss, losses, unclipped_grads
