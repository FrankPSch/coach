from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class BitFlip_DQN_HER(BitFlip_DQN):
    def __init__(self):
        super().__init__()
        self.memory = 'HindsightExperienceReplay'
        self.algorithm.state_value_to_use_as_goal = GoalTypes.Observation

        self.algorithm.hindsight_experience_replay_goal_selection_method = 'final'
        self.algorithm.hindsight_experience_replay_hindsight_transitions_per_regular_transition = 4


class EnvParams(Mujoco):
    def __init__(self):
        super().__init__()


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
