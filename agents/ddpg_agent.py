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
from agents.actor_critic_agent import ActorCriticAgent
from configurations import InputTypes, OutputTypes, NetworkParameters, MiddlewareTypes, AlgorithmParameters, \
    AgentParameters, InputEmbedderParameters, EmbedderScheme
from exploration_policies.ou_process import OUProcessParameters
from architectures.network_wrapper import NetworkWrapper
from memories.episodic_experience_replay import EpisodicExperienceReplayParameters
from utils import Signal, force_list
from core_types import RunPhase, ActionInfo, EnvironmentSteps
from spaces import Box
import numpy as np
from architectures.network_wrapper import NetworkWrapper
import copy


class DDPGCriticNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        observation_embedder = InputEmbedderParameters()
        observation_embedder.use_batchnorm = True
        action_embedder = InputEmbedderParameters()
        action_embedder.embedder_scheme = EmbedderScheme.Shallow
        self.input_types = {'observation': observation_embedder, 'action': action_embedder}
        self.middleware_type = MiddlewareTypes.FC
        self.output_types = [OutputTypes.V]
        self.loss_weights = [1.0]
        self.rescale_gradient_from_head_by_factor = [1]
        self.optimizer_type = 'Adam'
        self.hidden_layers_activation_function = 'relu'
        self.batch_size = 64
        self.async_training = True
        self.learning_rate = 0.001
        self.create_target_network = True


class DDPGActorNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        observation_embedder = InputEmbedderParameters()
        observation_embedder.use_batchnorm = True
        self.input_types = {'observation': observation_embedder}
        self.middleware_type = MiddlewareTypes.FC
        self.output_types = [OutputTypes.Pi]
        self.loss_weights = [1.0]
        self.rescale_gradient_from_head_by_factor = [1]
        self.optimizer_type = 'Adam'
        self.hidden_layers_activation_function = 'relu'
        self.batch_size = 64
        self.async_training = True
        self.learning_rate = 0.0001
        self.create_target_network = True


class DDPGAlgorithmParameters(AlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(1)
        self.rate_for_copying_weights_to_target = 0.001
        self.num_consecutive_playing_steps = EnvironmentSteps(1)


class DDPGAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=DDPGAlgorithmParameters(),
                         exploration=OUProcessParameters(),
                         memory=EpisodicExperienceReplayParameters(),
                         networks={"actor": DDPGActorNetworkParameters(), "critic": DDPGCriticNetworkParameters()})

    @property
    def path(self):
        return 'agents.ddpg_agent:DDPGAgent'


# Deep Deterministic Policy Gradients Network - https://arxiv.org/pdf/1509.02971.pdf
class DDPGAgent(ActorCriticAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)

        self.q_values = self.register_signal("Q")

    def learn_from_batch(self, batch):
        current_states, next_states, actions, rewards, game_overs, _ = self.extract_batch(batch, 'actor')

        # TD error = r + discount*max(q_st_plus_1) - q_st
        next_actions = self.networks['actor'].target_network.predict(next_states)
        inputs = copy.copy(next_states)
        inputs['action'] = next_actions
        q_st_plus_1 = self.networks['critic'].target_network.predict(inputs)
        TD_targets = np.expand_dims(rewards, -1) + \
                     (1.0 - np.expand_dims(game_overs, -1)) * self.ap.algorithm.discount * q_st_plus_1

        # get the gradients of the critic output with respect to the action
        actions_mean = self.networks['actor'].online_network.predict(current_states)
        critic_online_network = self.networks['critic'].online_network
        # TODO: convert into call to predict, current method ignores lstm middleware for example
        action_gradients = self.networks['critic'].sess.run(critic_online_network.gradients_wrt_inputs[0]['action'],
                                                        feed_dict=critic_online_network._feed_dict({
                                                            **current_states,
                                                            'action': actions_mean,
                                                        }))[0]

        # train the critic
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, -1)
        result = self.networks['critic'].train_and_sync_networks({**current_states, 'action': actions}, TD_targets)
        total_loss, losses, unclipped_grads = result[:3]

        # apply the gradients from the critic to the actor
        actor_online_network = self.networks['actor'].online_network
        gradients = self.networks['actor'].sess.run(actor_online_network.weighted_gradients[0],
                                                feed_dict=actor_online_network._feed_dict({
                                                    **current_states,
                                                    actor_online_network.gradients_weights_ph[0]: -action_gradients,
                                                }))
        if self.networks['actor'].has_global:
            self.networks['actor'].global_network.apply_gradients(gradients)
            self.networks['actor'].update_online_network()
        else:
            self.networks['actor'].online_network.apply_gradients(gradients)

        return total_loss, losses, unclipped_grads

    def train(self):
        return Agent.train(self)

    def choose_action(self, curr_state):
        # TODO: shouldn't this function be inherited?
        if type(self.spaces.action) != Box:
            raise ValueError("DDPG works only for continuous control problems")
        # convert to batch so we can run it through the network
        tf_input_state = self.dict_state_to_batches_dict(curr_state, 'actor')
        actor_network = self.networks['actor'].online_network
        actor_network.set_variable_value(
            assign_op=actor_network.output_heads[0].policy_std,
            value=self.exploration_policy.get_control_param(),  # TODO: make this work with additive noise etc.
            placeholder=actor_network.output_heads[0].policy_std_placeholder
        )
        action_values = actor_network.predict(tf_input_state).squeeze()

        action = self.exploration_policy.get_action(action_values)

        # bound actions
        action = self.spaces.action.clip_action_to_space(action)

        # get q value
        tf_input_state = self.dict_state_to_batches_dict(curr_state, 'critic')
        action_batch = np.expand_dims(action, 0)
        if type(action) != np.ndarray:
            action_batch = np.array([[action]])
        tf_input_state['action'] = action_batch
        q_value = self.networks['critic'].online_network.predict(tf_input_state)[0]
        self.q_values.add_sample(q_value)

        action_info = ActionInfo(action=action,
                                 action_value=q_value)
        return action_info
