from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, DDPG, OUExploration, None)
        self.algorithm.embedder_depth = EmbedderDepth.Deep
        self.learning_rate = 0.0001
        self.num_heatup_steps = 1000
        self.algorithm.num_consecutive_training_steps = 5


class EnvParams(Carla):
    def __init__(self):
        super().__init__()


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
