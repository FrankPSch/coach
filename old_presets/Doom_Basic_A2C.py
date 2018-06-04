from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, ActorCritic, CategoricalExploration, None)
        pass
        self.algorithm.policy_gradient_rescaler = 'A_VALUE'
        self.learning_rate = 0.00025
        self.num_heatup_steps = 100
        pass


class EnvParams(Doom):
    def __init__(self):
        super().__init__()
        self.level = 'basic'
        self.reward_scaling = 100.


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
