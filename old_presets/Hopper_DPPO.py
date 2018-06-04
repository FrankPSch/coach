from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, PPO, ExplorationParameters, None)
        pass
        self.learning_rate = 0.00001
        self.num_heatup_steps = 0
        self.algorithm.num_consecutive_training_steps = 1
        self.algorithm.num_consecutive_playing_steps = 5000
        self.algorithm.discount = 0.99
        self.batch_size = 128
        self.algorithm.policy_gradient_rescaler = 'GAE'
        self.algorithm.gae_lambda = 0.96
        pass
        self.algorithm.optimizer_type = 'Adam'
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
        self.dump_csv = True

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
