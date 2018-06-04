from filters.action.action_filter import ActionFilter
from typing import Union
import numpy as np
from spaces import Box, ActionSpace
from core_types import ActionType


class LinearBoxToBoxMap(ActionFilter):
    """
    Maps a box action space to a box action space.
    For example,
    - the source action space has actions of shape 1 with values between -42 and -10,
    - the target action space has actions of shape 1 with values between 10 and 32
    The mapping will add an offset of 52 to the incoming actions and then multiply them by 22/32 to scale them to the
    target action space
    The shape of the source and target action spaces is always the same
    """
    def __init__(self,
                 input_space_low: Union[None, int, float, np.ndarray],
                 input_space_high: Union[None, int, float, np.ndarray]):
        self.input_space_low = input_space_low
        self.input_space_high = input_space_high
        self.rescale = None
        self.offset = None
        super().__init__()

    def validate_output_action_space(self, output_action_space: Box):
        if not isinstance(output_action_space, Box):
            raise ValueError("Box discretization only works with an output space of type Box. "
                             "The given output space is {}".format(output_action_space))

    def get_unfiltered_action_space(self, output_action_space: Box) -> Box:
        self.input_action_space = Box(output_action_space.shape, self.input_space_low, self.input_space_high)
        self.rescale = \
            (output_action_space.high - output_action_space.low) / (self.input_space_high - self.input_space_low)
        self.offset = output_action_space.low - self.input_space_low
        self.output_action_space = output_action_space
        return self.input_action_space

    def filter(self, action: ActionType) -> ActionType:
        return self.output_action_space.low + (action - self.input_space_low) * self.rescale

