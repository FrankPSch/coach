from filters.filter import Filter
from spaces import ActionSpace
from core_types import ActionType

# class ActionAdapter(object):
#     """
#     Maps an action space to another action space. This is useful when the action space of the environment (simulator)
#     needs to be different than the action space of the agent.
#     """
#
#     def __init__(self, source: ActionSpace, target: ActionSpace):
#         self.source = source
#         self.target = target
#
#     def transform(self, action: ActionType) -> ActionType:
#         raise NotImplementedError("")
#
#     def __getattr__(self, item):
#         """
#         This will pass every requested function call to the source object if it is available there
#         """
#         available_attr = dir(ActionSpace)
#         if 'source' in self.__dict__:
#             available_attr += list(self.__dict__['source'].__dict__.keys())
#         if item in available_attr:
#             return getattr(self.__dict__['source'], item)
#         else:
#             return object.__getattribute__(self, item)


class ActionFilter(Filter):
    def __init__(self, input_action_space: ActionSpace=None):
        self.input_action_space = input_action_space
        self.output_action_space = None
        super().__init__()

    def get_unfiltered_action_space(self, output_action_space: ActionSpace) -> ActionSpace:
        """
        This function should contain the logic for getting the unfiltered action space
        :param output_action_space: the output action space
        :return: the unfiltered action space
        """
        return output_action_space

    def validate_output_action_space(self, output_action_space: ActionSpace):
        """
        A function that implements validation of the output action space
        :param output_action_space: the input action space
        :return: None
        """
        pass

    def validate_input_action(self, action: ActionType):
        """
        A function that verifies that the given action is in the expected input action space
        :param action: an action to validate
        :return: None
        """
        if not self.input_action_space.val_matches_space_definition(action):
            raise ValueError("The given action ({}) does not match the action space ({})"
                             .format(action, self.input_action_space))

    def validate_output_action(self, action: ActionType):
        """
        A function that verifies that the given action is in the expected output action space
        :param action: an action to validate
        :return: None
        """
        if not self.output_action_space.val_matches_space_definition(action):
            raise ValueError("The given action ({}) does not match the action space ({})"
                             .format(action, self.output_action_space))

    def filter(self, action: ActionType) -> ActionType:
        """
        A function that transforms from the agent's action space to the environment's action space
        :param action: an action to transform
        :return: transformed action
        """
        raise NotImplementedError("")

    def reverse_filter(self, action: ActionType) -> ActionType:
        """
        A function that transforms from the environment's action space to the agent's action space
        :param action: an action to transform
        :return: transformed action
        """
        raise NotImplementedError("")