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

from typing import Union, List, Dict

from core_types import EnvResponse, ActionInfo, RunPhase, StateType, ActionType, PredictionType
import numpy as np

from utils import force_list


class AgentInterface(object):
    def __init__(self):
        self._phase = RunPhase.HEATUP
        self.spaces = None

    @property
    def phase(self) -> RunPhase:
        """
        Get the phase of the agent
        :return: the current phase
        """
        return self._phase

    @phase.setter
    def phase(self, val: RunPhase):
        """
        Change the phase of the agent
        :param val: the new phase
        :return: None
        """
        self._phase = val

    def reset(self) -> None:
        """
        Reset the episode parameters for the agent
        :return: None
        """
        raise NotImplementedError("")

    def train(self) -> Union[float, List]:
        """
        Train the agents network
        :return: The loss of the training
        """
        raise NotImplementedError("")

    def act(self) -> ActionInfo:
        """
        Get a decision of the next action to take.
        The action is dependent on the current state which the agent holds from resetting the environment or from
        the observe function.
        :return: A tuple containing the actual action and additional info on the action
        """
        raise NotImplementedError("")

    def observe(self, env_response: EnvResponse) -> bool:
        """
        Gets a response from the environment.
        Processes this information for later use. For example, create a transition and store it in memory.
        The action info (a class containing any info the agent wants to store regarding its action decision process) is
        stored by the agent itself when deciding on the action.
        :param env_response: a EnvResponse containing the response from the environment
        :return: a done signal which is based on the agent knowledge. This can be different from the done signal from
                 the environment. For example, an agent can decide to finish the episode each time it gets some
                 intrinsic reward
        """
        raise NotImplementedError("")

    def save_model(self, model_id: str) -> None:
        """
        Save the model of the agent to the disk. This can contain the network parameters, the memory of the agent, etc.
        :param model_id: the model if to use for saving
        :return: None
        """
        raise NotImplementedError("")

    def get_predictions(self, states: Dict, prediction_type: PredictionType) -> np.ndarray:
        """
        Get a prediction from the agent with regard to the requested prediction_type. If the agent cannot predict this
        type of prediction_type, or if there is more than possible way to do so, raise a ValueException.
        :param states:
        :param prediction_type:
        :return: the agent's prediction
        """
        raise NotImplementedError("")



    #
    # def get_v_values_for_states(self, states: Union[StateType, List[StateType]]) -> np.ndarray:
    #     """
    #     Get a V value prediction for a batch of states. Agents that do not have a value prediction should raise a
    #      ValueError.
    #     :param states:
    #     :return:  a list of values for each state in the batch
    #     """
    #
    #     cannot_predict_v_error = ValueError("Cannot predict a V value, as the agent does not have a V head or a Q head")
    #     states = force_list(states)
    #
    #     if not self.can_predict_v():
    #         if not self.can_predict_q():
    #             raise cannot_predict_v_error
    #
    #         q_values = self.get_all_q_values_for_states(states) # q_values is now a list of np.ndarray, with q_value per
    #                                                             #  action
    #         return np.mean(q_values, axis=1)
    #
    # def get_q_values_for_action_state_pairs(self, actions: ActionType,
    #                                         states: Union[StateType, List[StateType]]) -> Union[float, List[float]]:
    #     """
    #     Get a Q value prediction for a batch of action-state pairs. Agents that do not have a Q value prediction should
    #      raise a ValueError.
    #     :param actions:
    #     :param states:
    #     :return:  a list of values for each state in the batch
    #     """
    #
    #     pass
    #
    # def get_all_q_values_for_states(self, states: StateType):
    #     if not self.can_predict_q():
    #         raise ValueError("This agent type cannot predict Q values. ")
    #
    # def get_action_probabilities_for_states(self, states: StateType) -> np.ndarray:
    #     if not self.can_predict_action_probabilities():
    #         raise ValueError("This agent type cannot predict action probabilities. ")
    #
    # def can_predict_v(self):
    #     return False
    #
    # def can_predict_q(self):
    #     return False
    #
    # def can_predict_action_probabilities(self):
    #     return False
