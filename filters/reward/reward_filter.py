from filters.filter import Filter
from spaces import RewardSpace


class RewardFilter(Filter):
    def __init__(self):
        super().__init__()

    def get_filtered_reward_space(self, input_reward_space: RewardSpace) -> RewardSpace:
        """
        This function should contain the logic for getting the filtered reward space
        :param input_reward_space: the input reward space
        :return: the filtered reward space
        """
        return input_reward_space