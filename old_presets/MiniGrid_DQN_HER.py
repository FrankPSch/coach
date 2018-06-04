from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, DQN, ExplorationParameters, None)
        pass
        self.learning_rate = 0.001

        self.input_types = [InputEmbedderParameters(), InputEmbedderParameters()]
        self.memory = 'HindsightExperienceReplay'
        self.state_values_to_use = ['observation', 'goal']
        self.exploration.epsilon_decay_steps = 5000


class EnvParams(Mujoco):
    def __init__(self):
        super().__init__()
        self.level = 'gym_minigrids:RandomGoalEnv'


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
