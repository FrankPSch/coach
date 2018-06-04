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
from logger import screen
from core_types import RunPhase, EnvironmentSteps
import os
import pickle
import numpy as np
from configurations import AlgorithmParameters, InputTypes, OutputTypes, NetworkParameters, AgentParameters, \
    MiddlewareTypes, InputEmbedderParameters
from exploration_policies.e_greedy import EGreedyParameters
from memories.episodic_experience_replay import EpisodicExperienceReplayParameters, MemoryGranularity
from schedules import ConstantSchedule


class NECNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_types = {'observation': InputEmbedderParameters()}
        self.middleware_type = MiddlewareTypes.FC
        self.output_types = [OutputTypes.DNDQ]
        self.loss_weights = [1.0]
        self.rescale_gradient_from_head_by_factor = [1]
        self.optimizer_type = 'Adam'
        self.input_rescaler = 1.0


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


class NECMemoryParameters(EpisodicExperienceReplayParameters):
    def __init__(self):
        super().__init__()
        self.n_step = 100
        self.discount = 0.99
        self.bootstrap_total_return_from_old_policy = True
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

    def learn_from_batch(self, batch):
        if not self.networks['main'].online_network.output_heads[0].DND.has_enough_entries(self.ap.algorithm.number_of_knn):
            return 0, [], 0
        else:
            if not self.training_started:
                self.training_started = True
                screen.log_title("Finished collecting initial entries in DND. Starting to train network...")

        current_states, next_states, actions, rewards, game_overs, total_return = self.extract_batch(batch, 'main')

        TD_targets = self.networks['main'].online_network.predict(current_states)

        #  only update the action that we have actually done in this transition
        for i in range(self.ap.network_wrappers['main'].batch_size):
            TD_targets[i, actions[i]] = total_return[i]

        # set the gradients to fetch for the DND update
        fetches = []
        head = self.networks['main'].online_network.output_heads[0]
        if self.ap.algorithm.propagate_updates_to_DND:
            fetches = [head.dnd_embeddings_grad, head.dnd_values_grad, head.dnd_indices]

        # train the neural network
        result = self.networks['main'].train_and_sync_networks(current_states, TD_targets, fetches)

        total_loss, losses, unclipped_grads = result[:3]

        # update the DND keys and values using the extracted gradients
        if self.ap.algorithm.propagate_updates_to_DND:
            embedding_gradients = np.swapaxes(result[-1][0], 0, 1)
            value_gradients = np.swapaxes(result[-1][1], 0, 1)
            indices = np.swapaxes(result[-1][2], 0, 1)
            head.DND.update_keys_and_values(actions, embedding_gradients, value_gradients, indices)

        return total_loss, losses, unclipped_grads

    def act(self):
        if self._phase == RunPhase.HEATUP:
            # get embedding in heatup (otherwise we get it through choose_action)
            embedding = self.networks['main'].online_network.predict(
                self.dict_state_to_batches_dict(self.curr_state, 'main'),
                outputs=self.networks['main'].online_network.state_embedding)
            self.current_episode_state_embeddings.append(embedding)

        return super().act()

    def get_prediction(self, states):
        # get the actions q values and the state embedding
        embedding, actions_q_values = self.networks['main'].online_network.predict(
            self.dict_state_to_batches_dict(states, 'main'),
            outputs=[self.networks['main'].online_network.state_embedding,
                     self.networks['main'].online_network.output_heads[0].output]
        )

        # store the state embedding for inserting it to the DND later
        self.current_episode_state_embeddings.append(embedding.squeeze())
        actions_q_values = actions_q_values[0][0]
        return actions_q_values

    def reset(self):
        super().reset()
        self.current_episode_state_embeddings = []

    def end_episode(self):
        # get the last full episode that we have collected
        episode = self.memory.get_last_complete_episode()
        if episode is not None:
            # the indexing is only necessary because the heatup can end in the middle of an episode
            # this won't be required after fixing this so that when the heatup is ended, the episode is closed
            returns = episode.get_transitions_attribute('total_return')[:len(self.current_episode_state_embeddings)]
            actions = episode.get_transitions_attribute('action')[:len(self.current_episode_state_embeddings)]
            self.networks['main'].online_network.output_heads[0].DND.add(self.current_episode_state_embeddings,
                                                                     actions, returns)

        super().end_episode()

    def save_model(self, model_id):
        self.networks['main'].save_model(model_id)
        with open(os.path.join(self.ap.save_model_dir, str(model_id) + '.dnd'), 'wb') as f:
            pickle.dump(self.networks['main'].online_network.output_heads[0].DND, f, pickle.HIGHEST_PROTOCOL)