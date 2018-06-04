from typing import List, Union, Dict, Any
import numpy as np
from enum import Enum

ActionType = Union[int, float, np.ndarray, List]
GoalType = Union[None, np.ndarray]
ObservationType = np.ndarray
RewardType = Union[int, float, np.ndarray]
StateType = Dict[str, np.ndarray]


class GoalTypes(object):
    Embedding = 1
    EmbeddingChange = 2
    Observation = 3
    Measurements = 4


# step methods

class StepMethod(object):
    def __init__(self, num_steps: int):
        self._num_steps = self.num_steps = num_steps

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @num_steps.setter
    def num_steps(self, val: int) -> None:
        self._num_steps = val


class Frames(StepMethod):
    def __init__(self, num_steps):
        super().__init__(num_steps)


class EnvironmentSteps(StepMethod):
    def __init__(self, num_steps):
        super().__init__(num_steps)


class Episodes(StepMethod):
    def __init__(self, num_steps):
        super().__init__(num_steps)


class TrainingSteps(StepMethod):
    def __init__(self, num_steps):
        super().__init__(num_steps)


class Time(StepMethod):
    def __init__(self, num_steps):
        super().__init__(num_steps)


class PredictionType(object):
    pass


class VStateValue(PredictionType):
    pass


class QActionStateValue(PredictionType):
    pass


class ActionProbabilities(PredictionType):
    pass


class Embedding(PredictionType):
    pass


class InputEmbedding(Embedding):
    pass


class MiddlewareEmbedding(Embedding):
    pass


class InputImageEmbedding(InputEmbedding):
    pass


class InputVectorEmbedding(InputEmbedding):
    pass


class Middleware_FC_Embedding(MiddlewareEmbedding):
    pass


class Middleware_LSTM_Embedding(MiddlewareEmbedding):
    pass


class Measurements(PredictionType):
    pass

PlayingStepsType = Union[EnvironmentSteps, Episodes, Frames]


# run phases
class RunPhase(Enum):
    HEATUP = "Heatup"
    TRAIN = "Training"
    TEST = "Testing"
    UNDEFINED = "Undefined"


# transitions

class Transition(object):
    def __init__(self, state: Dict[str, np.ndarray]=None, action: ActionType=None, reward: RewardType=None,
                 next_state: Dict[str, np.ndarray]=None, game_over: bool=None, info: Dict=None, goal: np.ndarray=None):
        """
        A transition is a tuple containing the information of a single step of interaction
        between the agent and the environment. The most basic version should contain the following values:
        (current state, action, reward, next state, game over)
        For imitation learning algorithms, if the reward, next state or game over is not known,
        it is sufficient to store the current state and action taken by the expert.

        :param state: The current state. Assumed to be a dictionary where the observation
                      is located at state['observation']
        :param action: The current action that was taken
        :param reward: The reward received from the environment
        :param next_state: The next state of the environment after applying the action.
                           The next state should be similar to the state in its structure.
        :param game_over: A boolean which should be True if the episode terminated after
                          the execution of the action.
        :param info: A dictionary containing any additional information to be stored in the transition
        :param goal: A goal that was attached to the specific timestep. It can be a vector or an image, and
                     can be defined either by the agent or the environment.
        """

        self._state = self.state = state
        self._action = self.action = action
        self._reward = self.reward = reward
        self._total_return = self.total_return = 0
        if not next_state:
            next_state = state
        self._next_state = self._next_state = next_state
        self._game_over = self.game_over = game_over
        self._goal = self.goal = goal
        if info is None:
            self.info = {}
        else:
            self.info = info

    def __repr__(self):
        return str(self.__dict__)

    @property
    def state(self):
        if self._state is None:
            raise Exception("The state was not filled by any of the modules between the environment and the agent")
        return self._state

    @state.setter
    def state(self, val):
        self._state = val

    @property
    def action(self):
        if self._action is None:
            raise Exception("The action was not filled by any of the modules between the environment and the agent")
        return self._action

    @action.setter
    def action(self, val):
        self._action = val

    @property
    def reward(self):

        if self._reward is None:
            raise Exception("The reward was not filled by any of the modules between the environment and the agent")
        return self._reward

    @reward.setter
    def reward(self, val):
        self._reward = val

    @property
    def total_return(self):
        if self._total_return is None:
            raise Exception("The total_return was not filled by any of the modules between the environment and the "
                            "agent")
        return self._total_return

    @total_return.setter
    def total_return(self, val):
        self._total_return = val

    @property
    def game_over(self):
        if self._game_over is None:
            raise Exception("The done flag was not filled by any of the modules between the environment and the agent")
        return self._game_over

    @game_over.setter
    def game_over(self, val):
        self._game_over = val

    @property
    def next_state(self):
        if self._next_state is None:
            raise Exception("The next state was not filled by any of the modules between the environment and the agent")
        return self._next_state

    @next_state.setter
    def next_state(self, val):
        self._next_state = val

    @property
    def goal(self):
        if self._goal is None:
            raise Exception("The goal was not filled by any of the modules between the environment and the agent")
        return self._goal

    @goal.setter
    def goal(self, val):
        self._goal = val

    def add_info(self, new_info: Dict[str, Any]) -> None:
        if not new_info.keys().isdisjoint(self.info.keys()):
            raise ValueError("The new info dictionary can not be appended to the existing info dictionary since there "
                             "are overlapping keys between the two. old keys: {}, new keys: {}"
                             .format(self.info.keys(), new_info.keys()))
        self.info.update(new_info)


class EnvResponse(object):
    def __init__(self, new_state: Dict[str, ObservationType], reward: RewardType, game_over: bool, info: Dict=None,
                 goal: ObservationType=None):
        """
        An env response is a collection containing the information returning from the environment after a single action
        has been performed on it.
        :param new_state: The new state that the environment has transitioned into. Assumed to be a dictionary where the
                          observation is located at state['observation']
        :param reward: The reward received from the environment
        :param game_over: A boolean which should be True if the episode terminated after
                          the execution of the action.
        :param info: any additional info from the environment
        :param goal: a goal defined by the environment
        """
        self._new_state = self.new_state = new_state
        self._reward = self.reward = reward
        self._game_over = self.game_over = game_over
        self._goal = self.goal = goal
        if info is None:
            self.info = {}
        else:
            self.info = info

    def __repr__(self):
        return str(self.__dict__)

    @property
    def new_state(self):
        return self._new_state

    @new_state.setter
    def new_state(self, val):
        self._new_state = val

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, val):
        self._reward = val

    @property
    def game_over(self):
        return self._game_over

    @game_over.setter
    def game_over(self, val):
        self._game_over = val

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, val):
        self._goal = val

    def add_info(self, info: Dict[str, Any]) -> None:
        if info.keys().isdisjoint(self.info.keys()):
            raise ValueError("The new info dictionary can not be appended to the existing info dictionary since there"
                             "are overlapping keys between the two")
        self.info.update(info)


class ActionInfo(object):
    """
    Action info is a class that holds an action and various additional information details about it
    """
    def __init__(self, action: ActionType, action_probability: float=0,
                 action_value: float=0., state_value: float=0., max_action_value: float=None,
                 action_intrinsic_reward: float=0):
        """
        :param action: the action
        :param action_probability: the probability that the action was given when selecting it
        :param action_value: the state-action value (Q value) of the action
        :param state_value: the state value (V value) of the state where the action was taken
        :param max_action_value: in case this is an action that was selected randomly, this is the value of the action
                                 that received the maximum value. if no value is given, the action is assumed to be the
                                 action with the maximum value
        :param action_intrinsic_reward: can contain any intrinsic reward that the agent wants to add to this action
                                        selection
        """
        self.action = action
        self.action_probability = action_probability
        self.action_value = action_value
        self.state_value = state_value
        if not max_action_value:
            self.max_action_value = action_value
        else:
            self.max_action_value = max_action_value
        self.action_intrinsic_reward = action_intrinsic_reward
