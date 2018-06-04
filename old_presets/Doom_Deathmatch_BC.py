from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, BC, ExplorationParameters, None)
        pass
        self.algorithm.load_memory_from_file_path = 'datasets/doom_deathmatch.p'
        self.learning_rate = 0.0005
        self.num_heatup_steps = 0
        self.evaluation_episodes = 5
        self.batch_size = 120
        self.evaluate_every_x_training_iterations = 100


class EnvParams(Doom):
    def __init__(self):
        super().__init__()
        self.level = 'deathmatch'


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
