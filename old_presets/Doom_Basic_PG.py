from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, PolicyGradient, CategoricalExploration, None)
        pass
        self.algorithm.policy_gradient_rescaler = 'FUTURE_RETURN_NORMALIZED_BY_TIMESTEP'
        self.learning_rate = 0.00001
        self.num_heatup_steps = 0
        self.algorithm.beta_entropy = 0.01


class EnvParams(Doom):
    def __init__(self):
        super().__init__()
        self.level = 'basic'


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
