from filters.action.action_filter import ActionFilter
from spaces import Discrete, ActionSpace
from typing import List
from core_types import ActionType


class PartialDiscreteActionSpaceMap(ActionFilter):
    """
    Maps the given actions from the output space to discrete actions in the action space.
    For example, if there are 10 multiselect actions in the output space, the actions 0-9 will be mapped to those
    multiselect actions.
    """
    def __init__(self, target_actions: List[ActionType]=None, descriptions: List[str]=None):
        self.target_actions = target_actions
        self.descriptions = descriptions
        super().__init__()

    def validate_output_action_space(self, output_action_space: ActionSpace):
        if not self.target_actions:
            raise ValueError("The target actions were not set")
        for v in self.target_actions:
            if not output_action_space.val_matches_space_definition(v):
                raise ValueError("The values in the output actions ({}) do not match the output action "
                                 "space definition ({})".format(v, output_action_space))

    def get_unfiltered_action_space(self, output_action_space: ActionSpace) -> Discrete:
        self.output_action_space = output_action_space
        self.input_action_space = Discrete(len(self.target_actions), self.descriptions)
        return self.input_action_space

    def filter(self, action: ActionType) -> ActionType:
        return self.target_actions[action]

    def reverse_filter(self, action: ActionType) -> ActionType:
        return [(action == x).all() for x in self.target_actions].index(True)

