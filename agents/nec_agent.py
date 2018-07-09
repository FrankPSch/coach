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

import os
import pickle
from typing import Union

import numpy as np

from agents.value_optimization_agent import ValueOptimizationAgent
from architectures.tensorflow_components.heads.dnd_q_head import DNDQHeadParameters
from architectures.tensorflow_components.middlewares.fc_middleware import FCMiddlewareParameters
from base_parameters import AlgorithmParameters, NetworkParameters, AgentParameters, \
    InputEmbedderParameters
from core_types import RunPhase, EnvironmentSteps, Episode
from exploration_policies.e_greedy import EGreedyParameters
from logger import screen
from memories.episodic_experience_replay import EpisodicExperienceReplayParameters, MemoryGranularity
from schedules import ConstantSchedule


class NECNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters()}
        self.middleware_parameters = FCMiddlewareParameters()
        self.heads_parameters = [DNDQHeadParameters()]
        self.loss_weights = [1.0]
        self.rescale_gradient_from_head_by_factor = [1]
        self.optimizer_type = 'Adam'


class NECAlgorithmParameters(AlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.dnd_size = 500000
        self.l2_norm_added_delta = 0.001
        self.new_value_shift_coefficient = 0.1
        self.number_of_knn = 50
        self.DND_key_error_threshold = 0
        self.num_consecutive_playing_steps = EnvironmentSteps(4)
        self.propagate_updates_to_DND = False
        self.n_step = 100
        self.bootstrap_total_return_from_old_policy = True


class NECMemoryParameters(EpisodicExperienceReplayParameters):
    def __init__(self):
        super().__init__()
        self.max_size = (MemoryGranularity.Transitions, 100000)


class NECAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=NECAlgorithmParameters(),
                         exploration=EGreedyParameters(),
                         memory=NECMemoryParameters(),
                         networks={"main": NECNetworkParameters()})
        self.exploration.epsilon_schedule = ConstantSchedule(0.1)
        self.exploration.evaluation_epsilon = 0.01

    @property
    def path(self):
        return 'agents.nec_agent:NECAgent'


# Neural Episodic Control - https://arxiv.org/pdf/1703.01988.pdf
class NECAgent(ValueOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.current_episode_state_embeddings = []
        self.training_started = False
        self.current_episode_buffer = \
            Episode(discount=self.ap.algorithm.discount,
                    n_step=self.ap.algorithm.n_step,
                    bootstrap_total_return_from_old_policy=self.ap.algorithm.bootstrap_total_return_from_old_policy)

    def learn_from_batch(self, batch):
        if not self.networks['main'].online_network.output_heads[0].DND.has_enough_entries(self.ap.algorithm.number_of_knn):
            return 0, [], 0
        else:
            if not self.training_started:
                self.training_started = True
                screen.log_title("Finished collecting initial entries in DND. Starting to train network...")

        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        TD_targets = self.networks['main'].online_network.predict(batch.states(network_keys))

        #  only update the action that we have actually done in this transition
        for i in range(self.ap.network_wrappers['main'].batch_size):
            TD_targets[i, batch.actions()[i]] = batch.total_returns()[i]

        # set the gradients to fetch for the DND update
        fetches = []
        head = self.networks['main'].online_network.output_heads[0]
        if self.ap.algorithm.propagate_updates_to_DND:
            fetches = [head.dnd_embeddings_grad, head.dnd_values_grad, head.dnd_indices]

        # train the neural network
        result = self.networks['main'].train_and_sync_networks(batch.states(network_keys), TD_targets, fetches)

        total_loss, losses, unclipped_grads = result[:3]

        # update the DND keys and values using the extracted gradients
        if self.ap.algorithm.propagate_updates_to_DND:
            embedding_gradients = np.swapaxes(result[-1][0], 0, 1)
            value_gradients = np.swapaxes(result[-1][1], 0, 1)
            indices = np.swapaxes(result[-1][2], 0, 1)
            head.DND.update_keys_and_values(batch.actions(), embedding_gradients, value_gradients, indices)

        return total_loss, losses, unclipped_grads

    def act(self):
        if self._phase == RunPhase.HEATUP:
            # get embedding in heatup (otherwise we get it through choose_action)
            embedding = self.networks['main'].online_network.predict(
                self.prepare_batch_for_inference(self.curr_state, 'main'),
                outputs=self.networks['main'].online_network.state_embedding)
            self.current_episode_state_embeddings.append(embedding)

        return super().act()

    def get_prediction(self, states):
        # get the actions q values and the state embedding
        embedding, actions_q_values = self.networks['main'].online_network.predict(
            self.prepare_batch_for_inference(states, 'main'),
            outputs=[self.networks['main'].online_network.state_embedding,
                     self.networks['main'].online_network.output_heads[0].output]
        )

        # store the state embedding for inserting it to the DND later
        self.current_episode_state_embeddings.append(embedding.squeeze())
        actions_q_values = actions_q_values[0][0]
        return actions_q_values

    def reset_internal_state(self):
        super().reset_internal_state()
        self.current_episode_state_embeddings = []
        self.current_episode_buffer = \
            Episode(discount=self.ap.algorithm.discount,
                    n_step=self.ap.algorithm.n_step,
                    bootstrap_total_return_from_old_policy=self.ap.algorithm.bootstrap_total_return_from_old_policy)

    def handle_episode_ended(self):
        # get the last full episode that we have collected
        episode = self.call_memory('get_last_complete_episode')
        if episode is not None:
            # the indexing is only necessary because the heatup can end in the middle of an episode
            # this won't be required after fixing this so that when the heatup is ended, the episode is closed
            returns = episode.get_transitions_attribute('total_return')[:len(self.current_episode_state_embeddings)]
            actions = episode.get_transitions_attribute('action')[:len(self.current_episode_state_embeddings)]
            self.networks['main'].online_network.output_heads[0].DND.add(self.current_episode_state_embeddings,
                                                                     actions, returns)

        super().handle_episode_ended()

    def save_checkpoint(self, checkpoint_id):
        with open(os.path.join(self.ap.task_parameters.save_checkpoint_dir, str(checkpoint_id) + '.dnd'), 'wb') as f:
            pickle.dump(self.networks['main'].online_network.output_heads[0].DND, f, pickle.HIGHEST_PROTOCOL)
