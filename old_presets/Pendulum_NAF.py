from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, NAF, AdditiveNoiseExploration, None)
        pass
        self.learning_rate = 0.001
        self.num_heatup_steps = 1000
        self.batch_size = 100
        pass

        self.test = True
        self.test_max_step_threshold = 100
        self.test_min_return_threshold = -250


class EnvParams(Mujoco):
    def __init__(self):
        super().__init__()
        self.level = 'Pendulum-v0'
        # self.reward_scaling = 1000


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
