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
import numpy as np
from core_types import ActionInfo
from configurations import AlgorithmParameters, AgentParameters, NetworkParameters, OutputTypes, \
    MiddlewareTypes, InputEmbedderParameters
from memories.episodic_experience_replay import EpisodicExperienceReplayParameters
from exploration_policies.e_greedy import EGreedyParameters
from spaces import SpacesDefinition, MeasurementsObservationSpace


class DFPNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_types = {'observation': InputEmbedderParameters(),
                            'measurements': InputEmbedderParameters(),
                            'goal': InputEmbedderParameters()}
        self.middleware_type = MiddlewareTypes.FC
        self.output_types = [OutputTypes.MeasurementsPrediction]
        self.loss_weights = [1.0]
        self.async_training = True


class DFPMemoryParameters(EpisodicExperienceReplayParameters):
    def __init__(self):
        super().__init__()
        self.num_predicted_steps_ahead = 6


class DFPAlgorithmParameters(AlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.num_predicted_steps_ahead = 6
        self.state_values_to_use = ['observation', 'measurements', 'goal']
        self.goal_vector = [1.0, 1.0]
        self.future_measurements_weights = [0.5, 0.5, 1.0]
        self.use_accumulated_reward_as_measurement = False


class DFPAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=DFPAlgorithmParameters(),
                         exploration=EGreedyParameters(),
                         memory=EpisodicExperienceReplayParameters(),
                         networks={"main": DFPNetworkParameters()})

    @property
    def path(self):
        return 'agents.dfp_agent:DFPAgent'


# Direct Future Prediction Agent - http://vladlen.info/papers/learning-to-act.pdf
class DFPAgent(Agent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.current_goal = self.ap.algorithm.goal_vector

    def learn_from_batch(self, batch):
        current_states, next_states, actions, rewards, game_overs, total_returns = self.extract_batch(batch, 'main')

        # get the current outputs of the network
        targets = self.networks['main'].online_network.predict(current_states)

        # change the targets for the taken actions
        for i in range(self.ap.network_wrappers['main'].batch_size):
            targets[i, actions[i]] = batch[i].info['future_measurements'].flatten()

        result = self.networks['main'].train_and_sync_networks(current_states, targets)
        total_loss, losses, unclipped_grads = result[:3]

        return total_loss, losses, unclipped_grads

    def choose_action(self, curr_state):
        # predict the future measurements
        tf_input_state = self.dict_state_to_batches_dict(curr_state, 'main')
        measurements_future_prediction = self.networks['main'].online_network.predict(tf_input_state)[0]
        action_values = np.zeros((self.spaces.action.shape,))
        num_steps_used_for_objective = len(self.ap.algorithm.future_measurements_weights)

        # calculate the score of each action by multiplying it's future measurements with the goal vector
        for action_idx in range(self.spaces.action.shape):
            action_measurements = measurements_future_prediction[action_idx]
            action_measurements = np.reshape(action_measurements,
                                             (self.ap.algorithm.num_predicted_steps_ahead, self.spaces.measurements.shape))
            future_steps_values = np.dot(action_measurements, self.current_goal)
            action_values[action_idx] = np.dot(future_steps_values[-num_steps_used_for_objective:],
                                               self.ap.algorithm.future_measurements_weights)

        # choose action according to the exploration policy and the current phase (evaluating or training the agent)
        action = self.exploration_policy.get_action(action_values)

        action_values = action_values.squeeze()

        action_info = ActionInfo(action=action, action_value=action_values[action])
        return action, action_info

    def set_environment_parameters(self, spaces: SpacesDefinition):
        super().set_environment_parameters(spaces)
        self.spaces.state['goal'] = MeasurementsObservationSpace(shape=self.spaces.state['measurements'].shape,
                                                                 low=self.spaces.state['measurements'].low,
                                                                 high=self.spaces.state['measurements'].high)


