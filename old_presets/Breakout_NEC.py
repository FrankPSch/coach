from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, NEC, ExplorationParameters, None)
        pass
        self.learning_rate = 0.00025
        self.algorithm.num_transitions_in_experience_replay = 1000000
        self.exploration.initial_epsilon = 1.0
        self.exploration.final_epsilon = 0.1
        self.exploration.epsilon_decay_steps = 1000000
        self.num_heatup_steps = 50000


class EnvParams(Atari):
    def __init__(self):
        super().__init__()
        self.level = 'BreakoutDeterministic-v4'


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
