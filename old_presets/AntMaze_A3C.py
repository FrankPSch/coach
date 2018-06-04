from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AntMaze_A3C(Ant_A3C):
    def __init__(self):
        Ant_A3C.__init__(self)
        pass



class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
