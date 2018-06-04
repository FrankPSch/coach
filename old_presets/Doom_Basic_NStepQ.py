from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, NStepQ, ExplorationParameters, None)
        pass
        self.learning_rate = 0.000025
        self.num_heatup_steps = 0
        self.algorithm.num_steps_between_copying_online_weights_to_target = 1000
        self.algorithm.optimizer_type = 'Adam'
        self.clip_gradients = 1000


class EnvParams(Doom):
    def __init__(self):
        super().__init__()
        self.level = 'basic'


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
