from filters.action.partial_discrete_action_space_map import PartialDiscreteActionSpaceMap
from spaces import ActionSpace, Discrete


class FullDiscreteActionSpaceMap(PartialDiscreteActionSpaceMap):
    """
    Maps all the actions in the output space to discrete actions in the action space.
    For example, if there are 10 multiselect actions in the output space, the actions 0-9 will be mapped to those
    multiselect actions.
    """
    def __init__(self):
        super().__init__()

    def get_unfiltered_action_space(self, output_action_space: ActionSpace) -> Discrete:
        self.target_actions = output_action_space.actions
        return super().get_unfiltered_action_space(output_action_space)
