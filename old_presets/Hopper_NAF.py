from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, NAF, AdditiveNoiseExploration, None)
        pass
        self.learning_rate = 0.0005
        self.num_heatup_steps = 1000
        self.batch_size = 100
        self.algorithm.async_training = True
        pass


class EnvParams(Mujoco):
    def __init__(self):
        super().__init__()
        self.level = 'Hopper-v1'
        self.normalize_observation = True


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
