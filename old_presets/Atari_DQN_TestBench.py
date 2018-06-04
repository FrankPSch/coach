from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, DQN, ExplorationParameters, None)
        pass
        self.algorithm.num_steps_between_copying_online_weights_to_target = 10000
        self.learning_rate = 0.00025
        self.algorithm.num_transitions_in_experience_replay = 1000000
        self.exploration.initial_epsilon = 1.0
        self.exploration.final_epsilon = 0.1
        self.exploration.epsilon_decay_steps = 1000000
        self.exploration.evaluation_policy = 'EGreedy'
        self.exploration.evaluation_epsilon = 0.05
        self.num_heatup_steps = 10000
        self.evaluation_episodes = 25
        self.evaluate_every_x_episodes = 1000
        self.num_training_iterations = 500


class EnvParams(Atari):
    def __init__(self):
        super().__init__()
        self.level = 'BreakoutDeterministic-v4'


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
