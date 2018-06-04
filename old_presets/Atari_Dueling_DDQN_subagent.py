from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, DDQN, EGreedyExploration, None)
        self.algorithm.output_types = [OutputTypes.DuelingQ]
        self.algorithm.num_episodes_in_experience_replay = 200
        self.learning_rate = 0.00025
        self.num_heatup_steps = 1000
        self.algorithm.num_playing_steps_between_two_training_steps = 1
        self.algorithm.ignore_extrinsic_reward = True
        self.algorithm.add_intrinsic_reward_for_reaching_the_goal = True
        self.algorithm.distance_from_goal_threshold = 0.1
        self.algorithm.state_value_to_use_as_goal = GoalTypes.Embedding
        self.algorithm.state_values_to_use = ['observation', 'goal']
        self.algorithm.input_types = [InputEmbedderParameters(), InputEmbedderParameters()]
        pass
        self.algorithm.num_consecutive_training_steps = 10


class EnvParams(Atari):
    def __init__(self):
        super().__init__()


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()
        self.print_summary = True

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
