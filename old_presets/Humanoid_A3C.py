from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, ActorCritic, EntropyExploration, None)
        pass
        self.algorithm.policy_gradient_rescaler = 'A_VALUE'
        self.algorithm.optimizer_type = 'Adam'
        self.learning_rate = 0.00002
        self.num_heatup_steps = 0
        pass
        self.algorithm.discount = 0.99
        self.algorithm.apply_gradients_every_x_episodes = 1
        self.algorithm.num_steps_between_gradient_updates = 20
        self.algorithm.gae_lambda = 1
        self.algorithm.beta_entropy = 0.005
        self.clip_gradients = 40
        self.algorithm.middleware_type = MiddlewareTypes.FC
        pass


class EnvParams(Mujoco):
    def __init__(self):
        super().__init__()
        self.level = 'Humanoid-v1'
        self.reward_scaling = 20.
        self.normalize_observation = True


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
