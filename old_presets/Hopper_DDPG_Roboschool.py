from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, DDPG, OUExploration, None)
        pass
        self.learning_rate = 0.00025
        self.num_heatup_steps = 100


class EnvParams(Roboschool):
    def __init__(self):
        super().__init__()
        self.level = 'RoboschoolHopper-v1'


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
