from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, DDPG, OUExploration, None)
        pass
        self.learning_rate = 0.00025
        self.num_heatup_steps = 1000
        pass


class EnvParams(Mujoco):
    def __init__(self):
        super().__init__()
        self.level = 'HalfCheetah-v1'
        self.normalize_observation = True


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
