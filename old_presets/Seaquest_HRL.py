from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, DQN, EGreedyExploration, None)
        pass
        self.subagent_timestep_limit = 2
        self.num_heatup_steps = 1000
        self.algorithm.num_playing_steps_between_two_training_steps = 1
        self.algorithm_parameters = [Atari_DDPG_subagent, Atari_Dueling_DDQN_subagent]


class EnvParams(Atari):
    def __init__(self):
        super().__init__()
        self.level = 'Seaquest-v0'


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
