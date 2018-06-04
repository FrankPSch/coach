from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, BootstrappedDQN, BootstrappedDQNExploration, None)
        pass
        self.algorithm.num_episodes_in_experience_replay = 200
        self.learning_rate = 0.00025
        self.algorithm.num_steps_between_copying_online_weights_to_target = 1000
        self.num_heatup_steps = 1000


class EnvParams(Doom):
    def __init__(self):
        super().__init__()
        self.level = 'deadly_corridor'


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
