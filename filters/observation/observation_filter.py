from filters.filter import Filter
from spaces import ObservationSpace


class ObservationFilter(Filter):
    def __init__(self):
        super().__init__()

    def get_filtered_observation_space(self, input_observation_space: ObservationSpace) -> ObservationSpace:
        """
        This function should contain the logic for getting the filtered observation space
        :param input_observation_space: the input observation space
        :return: the filtered observation space
        """
        return input_observation_space

    def validate_input_observation_space(self, input_observation_space: ObservationSpace):
        """
        A function that implements validation of the input observation space
        :param input_observation_space: the input observation space
        :return: None
        """
        pass