from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, PPO, CategoricalExploration, None)
        pass
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.algorithm.num_consecutive_training_steps = 1
        self.algorithm.num_consecutive_playing_steps = 512
        self.algorithm.discount = 0.99
        self.batch_size = 128
        self.algorithm.policy_gradient_rescaler = 'A_VALUE'
        self.algorithm.optimizer_type = 'LBFGS'
        pass

        self.test = True
        self.test_max_step_threshold = 200
        self.test_min_return_threshold = 150


class EnvParams(Mujoco):
    def __init__(self):
        super().__init__()
        self.level = 'CartPole-v0'
        self.normalize_observation = True


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
