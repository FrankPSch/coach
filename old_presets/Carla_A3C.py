from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, ActorCritic, EntropyExploration, None)
        self.algorithm.embedder_depth = EmbedderDepth.Deep
        self.algorithm.policy_gradient_rescaler = 'GAE'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        pass
        self.algorithm.discount = 0.99
        self.algorithm.apply_gradients_every_x_episodes = 1
        self.algorithm.num_steps_between_gradient_updates = 30
        self.algorithm.gae_lambda = 1
        self.algorithm.beta_entropy = 0.01
        self.clip_gradients = 40
        self.algorithm.middleware_type = MiddlewareTypes.FC


class EnvParams(Carla):
    def __init__(self):
        super().__init__()
        # self.reward_scaling = 1.0e9


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
