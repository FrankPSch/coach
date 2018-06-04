from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, PolicyGradient, CategoricalExploration, None)
        pass
        self.algorithm.policy_gradient_rescaler = 'FUTURE_RETURN_NORMALIZED_BY_TIMESTEP'
        self.learning_rate = 0.001
        self.num_heatup_steps = 100
        self.algorithm.discount = 1.0

        self.test = True
        self.test_max_step_threshold = 150
        self.test_min_return_threshold = 150


class EnvParams(Mujoco):
    def __init__(self):
        super().__init__()
        self.level = 'CartPole-v0'


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
