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
from agents.policy_optimization_agent import PolicyOptimizationAgent
from utils import Signal, last_sample
import numpy as np
from configurations import AlgorithmParameters, AgentParameters, InputTypes, OutputTypes, MiddlewareTypes, \
    NetworkParameters, InputEmbedderParameters
from exploration_policies.e_greedy import EGreedyParameters
from memories.single_episode_buffer import SingleEpisodeBufferParameters


class NStepQNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_types = {'observation': InputEmbedderParameters()}
        self.middleware_type = MiddlewareTypes.FC
        self.output_types = [OutputTypes.Q]
        self.loss_weights = [1.0]
        self.optimizer_type = 'Adam'
        self.async_training = True
        self.shared_optimizer = True
        self.hidden_layers_activation_function = 'elu'
        self.create_target_network = True


class NStepQAlgorithmParameters(AlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.num_steps_between_copying_online_weights_to_target = 1000
        self.num_episodes_in_experience_replay = 2
        self.apply_gradients_every_x_episodes = 1
        self.num_steps_between_gradient_updates = 20  # this is called t_max in all the papers
        self.targets_horizon = 'N-Step'


class NStepQAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=NStepQAlgorithmParameters(),
                         exploration=EGreedyParameters(),
                         memory=SingleEpisodeBufferParameters(),
                         networks={"main": NStepQNetworkParameters()})

    @property
    def path(self):
        return 'agents.n_step_q_agent:NStepQAgent'


# N Step Q Learning Agent - https://arxiv.org/abs/1602.01783
class NStepQAgent(ValueOptimizationAgent, PolicyOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.last_gradient_update_step_idx = 0
        self.q_values = self.register_signal('Q Values')
        self.value_loss = self.register_signal('Value Loss')

    def learn_from_batch(self, batch):
        # batch contains a list of episodes to learn from
        current_states, next_states, actions, rewards, game_overs, _ = self.extract_batch(batch, 'main')

        # get the values for the current states
        state_value_head_targets = self.networks['main'].online_network.predict(current_states)

        # the targets for the state value estimator
        num_transitions = len(game_overs)

        if self.ap.algorithm.targets_horizon == '1-Step':
            # 1-Step Q learning
            q_st_plus_1 = self.networks['main'].target_network.predict(next_states)

            for i in reversed(range(num_transitions)):
                state_value_head_targets[i][actions[i]] = \
                    rewards[i] + (1.0 - game_overs[i]) * self.ap.algorithm.discount * np.max(q_st_plus_1[i], 0)

        elif self.ap.algorithm.targets_horizon == 'N-Step':
            # N-Step Q learning
            if game_overs[-1]:
                R = 0
            else:
                R = np.max(self.networks['main'].target_network.predict(last_sample(next_states)))

            for i in reversed(range(num_transitions)):
                R = rewards[i] + self.ap.algorithm.discount * R
                state_value_head_targets[i][actions[i]] = R

        else:
            assert True, 'The available values for targets_horizon are: 1-Step, N-Step'

        # train
        result = self.networks['main'].online_network.accumulate_gradients(current_states, [state_value_head_targets])

        # logging
        total_loss, losses, unclipped_grads = result[:3]
        self.value_loss.add_sample(losses[0])

        return total_loss, losses, unclipped_grads

    def train(self):
        # update the target network of every network that has a target network
        if self.total_steps_counter % self.ap.algorithm.num_steps_between_copying_online_weights_to_target == 0:
            for network in self.networks.values():
                network.update_target_network(self.ap.algorithm.rate_for_copying_weights_to_target)
            self.agent_logger.create_signal_value('Update Target Network', 1)
        else:
            self.agent_logger.create_signal_value('Update Target Network', 0, overwrite=False)

        return PolicyOptimizationAgent.train(self)
