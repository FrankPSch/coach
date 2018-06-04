from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, NEC, ExplorationParameters, None)
        pass
        self.learning_rate = 0.00025
        self.algorithm.num_episodes_in_experience_replay = 200
        self.num_heatup_steps = 1000
        self.exploration.epsilon_decay_steps = 1000
        self.exploration.final_epsilon = 0.1
        self.algorithm.discount = 1.0

        self.test = True
        self.test_max_step_threshold = 200
        self.test_min_return_threshold = 150


class EnvParams(Mujoco):
    def __init__(self):
        super().__init__()
        self.level = 'CartPole-v0'


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
