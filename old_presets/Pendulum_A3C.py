from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, ActorCritic, EntropyExploration, None)
        pass
        self.algorithm.policy_gradient_rescaler = 'GAE'
        self.algorithm.optimizer_type = 'Adam'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.algorithm.discount = 0.99
        self.algorithm.num_steps_between_gradient_updates = 5
        self.algorithm.gae_lambda = 1


class EnvParams(Mujoco):
    def __init__(self):
        super().__init__()
        self.level = 'Pendulum-v0'


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
