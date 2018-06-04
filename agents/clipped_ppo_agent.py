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

from agents.actor_critic_agent import ActorCriticAgent
from agents.policy_optimization_agent import PolicyGradientRescaler
from random import shuffle
import numpy as np

from block_factories.block_factory import DistributedTaskParameters
from utils import Signal
from logger import screen
from collections import OrderedDict
from core_types import RunPhase, ActionInfo, EnvironmentSteps
from memories.episodic_experience_replay import EpisodicExperienceReplayParameters
from exploration_policies.additive_noise import AdditiveNoiseParameters
from spaces import Discrete, Box
from configurations import AlgorithmParameters, InputTypes, OutputTypes, MiddlewareTypes, NetworkParameters, \
    AgentParameters, InputEmbedderParameters
import copy
from architectures.network_wrapper import NetworkWrapper


class ClippedPPONetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_types = {'observation': InputEmbedderParameters()}
        self.middleware_type = MiddlewareTypes.FC
        self.output_types = [OutputTypes.V, OutputTypes.PPO]
        self.loss_weights = [0.5, 1.0]
        self.rescale_gradient_from_head_by_factor = [1, 1]
        self.hidden_layers_activation_function = 'tanh'
        self.batch_size = 64
        self.optimizer_type = 'Adam'
        self.clip_gradients = 40
        self.use_separate_networks_per_head = True
        self.async_training = False
        self.l2_regularization = 1e-3
        self.create_target_network = True


class ClippedPPOAlgorithmParameters(AlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.num_episodes_in_experience_replay = 1000000
        self.policy_gradient_rescaler = PolicyGradientRescaler.GAE
        self.gae_lambda = 0.95
        self.use_kl_regularization = False
        self.add_a_normalized_timestep_to_the_observation = False
        self.value_targets_mix_fraction = 0.1
        self.clip_likelihood_ratio_using_epsilon = 0.2
        self.estimate_state_value_using_gae = True
        self.step_until_collecting_full_episodes = True
        self.beta_entropy = 0.01  # should be 0 for mujoco
        self.num_consecutive_playing_steps = EnvironmentSteps(2048)


class ClippedPPOAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=ClippedPPOAlgorithmParameters(),
                         exploration=AdditiveNoiseParameters(),
                         memory=EpisodicExperienceReplayParameters(),
                         networks={"main": ClippedPPONetworkParameters()})

    @property
    def path(self):
        return 'agents.clipped_ppo_agent:ClippedPPOAgent'


# Clipped Proximal Policy Optimization - https://arxiv.org/abs/1707.06347
class ClippedPPOAgent(ActorCriticAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        # signals definition
        self.value_loss = self.register_signal('Value Loss')
        self.policy_loss = self.register_signal('Policy Loss')
        self.total_kl_divergence_during_training_process = 0.0
        self.unclipped_grads = self.register_signal('Grads (unclipped)')
        self.value_targets = self.register_signal('Value Targets')
        self.kl_divergence = self.register_signal('KL Divergence')

    def fill_advantages(self, batch):
        current_states, next_states, actions, rewards, game_overs, total_return = self.extract_batch(batch, 'main')

        current_state_values = self.networks['main'].online_network.predict(current_states)[0]
        current_state_values = current_state_values.squeeze()
        self.state_values.add_sample(current_state_values)

        # calculate advantages
        advantages = []
        value_targets = []
        if self.policy_gradient_rescaler == PolicyGradientRescaler.A_VALUE:
            advantages = total_return - current_state_values
        elif self.policy_gradient_rescaler == PolicyGradientRescaler.GAE:
            # get bootstraps
            episode_start_idx = 0
            advantages = np.array([])
            value_targets = np.array([])
            for idx, game_over in enumerate(game_overs):
                if game_over:
                    # get advantages for the rollout
                    value_bootstrapping = np.zeros((1,))
                    rollout_state_values = np.append(current_state_values[episode_start_idx:idx+1], value_bootstrapping)

                    rollout_advantages, gae_based_value_targets = \
                        self.get_general_advantage_estimation_values(rewards[episode_start_idx:idx+1],
                                                                     rollout_state_values)
                    episode_start_idx = idx + 1
                    advantages = np.append(advantages, rollout_advantages)
                    value_targets = np.append(value_targets, gae_based_value_targets)
        else:
            screen.warning("WARNING: The requested policy gradient rescaler is not available")

        # standardize
        advantages = (advantages - np.mean(advantages)) / np.std(advantages)

        for transition, advantage, value_target in zip(batch, advantages, value_targets):
            transition.info['advantage'] = advantage
            transition.info['gae_based_value_target'] = value_target

        self.action_advantages.add_sample(advantages)

    def train_network(self, dataset, epochs):
        loss = []
        for j in range(epochs):
            loss = {
                'total_loss': [],
                'policy_losses': [],
                'unclipped_grads': [],
                'fetch_result': []
            }
            shuffle(dataset)
            for i in range(int(len(dataset) / self.ap.network_wrappers['main'].batch_size)):
                batch = dataset[i * self.ap.network_wrappers['main'].batch_size:(i + 1) * self.ap.network_wrappers['main'].batch_size]
                current_states, _, actions, _, _, total_return = self.extract_batch(batch, 'main')

                advantages = np.array([t.info['advantage'] for t in batch])
                gae_based_value_targets = np.array([t.info['gae_based_value_target'] for t in batch])
                if not isinstance(self.spaces.action, Discrete) and len(actions.shape) == 1:
                    actions = np.expand_dims(actions, -1)

                # get old policy probabilities and distribution
                result = self.networks['main'].target_network.predict(current_states)
                old_policy_distribution = result[1:]

                # calculate gradients and apply on both the local policy network and on the global policy network
                fetches = [self.networks['main'].online_network.output_heads[1].kl_divergence,
                           self.networks['main'].online_network.output_heads[1].entropy]

                total_return = np.expand_dims(total_return, -1)
                value_targets = gae_based_value_targets if self.ap.algorithm.estimate_state_value_using_gae else total_return

                inputs = copy.copy(current_states)
                # TODO: why is this output 0 and not output 1?
                inputs['output_0_0'] = actions
                # TODO: does old_policy_distribution really need to be represented as a list?
                # A: yes it does, in the event of discrete controls, it has just a mean
                # otherwise, it has both a mean and standard deviation
                for input_index, input in enumerate(old_policy_distribution):
                    inputs['output_0_{}'.format(input_index + 1)] = input

                total_loss, policy_losses, unclipped_grads, fetch_result =\
                    self.networks['main'].online_network.accumulate_gradients(
                        inputs, [total_return, advantages], additional_fetches=fetches
                    )

                self.value_targets.add_sample(value_targets)
                if isinstance(self.ap.task_parameters, DistributedTaskParameters):
                    self.networks['main'].apply_gradients_to_global_network()
                    self.networks['main'].update_online_network()
                else:
                    self.networks['main'].apply_gradients_to_online_network()

                self.networks['main'].online_network.reset_accumulated_gradients()

                loss['total_loss'].append(total_loss)
                loss['policy_losses'].append(policy_losses)
                loss['unclipped_grads'].append(unclipped_grads)
                loss['fetch_result'].append(fetch_result)

                self.unclipped_grads.add_sample(unclipped_grads)

            for key in loss.keys():
                loss[key] = np.mean(loss[key], 0)

            if self.ap.network_wrappers['main'].learning_rate_decay_rate != 0:
                curr_learning_rate = self.networks['main'].online_network.get_variable_value(
                    self.ap.network_wrappers['main'].learning_rate)
                self.curr_learning_rate.add_sample(curr_learning_rate)
            else:
                curr_learning_rate = self.ap.network_wrappers['main'].learning_rate

            # log training parameters
            screen.log_dict(
                OrderedDict([
                    ("Surrogate loss", loss['policy_losses'][0]),
                    ("KL divergence", loss['fetch_result'][0]),
                    ("Entropy", loss['fetch_result'][1]),
                    ("training epoch", j),
                    ("learning_rate", curr_learning_rate)
                ]),
                prefix="Policy training"
            )

        self.total_kl_divergence_during_training_process = loss['fetch_result'][0]
        self.entropy.add_sample(loss['fetch_result'][1])
        self.kl_divergence.add_sample(loss['fetch_result'][0])
        return policy_losses

    def post_training_commands(self):

        # clean memory
        self.memory.clean()

    def train(self):
        loss = 0
        if self._should_train(wait_for_full_episode=True):
            for training_step in range(self.ap.algorithm.num_consecutive_training_steps):
                self.networks['main'].sync()

                dataset = self.memory.transitions

                self.fill_advantages(dataset)

                # take only the requested number of steps
                dataset = dataset[:self.ap.algorithm.num_consecutive_playing_steps.num_steps]

                losses = self.train_network(dataset, 10)

                self.value_loss.add_sample(losses[0])
                self.policy_loss.add_sample(losses[1])
                # TODO: pass the losses to the output of the function

            self.training_iteration += 1
            self.update_log()  # should be done in order to update the data that has been accumulated * while not playing *
            return np.append(losses[0], losses[1])
