from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, CategoricalDQN, ExplorationParameters, None)
        pass
        self.algorithm.num_steps_between_copying_online_weights_to_target = 100
        self.learning_rate = 0.00025
        self.algorithm.num_episodes_in_experience_replay = 200
        self.num_heatup_steps = 1000
        self.exploration.epsilon_decay_steps = 3000
        self.algorithm.discount = 1.0
        pass
        self.algorithm.v_min = 0.0
        self.algorithm.v_max = 200.0

        self.test = True
        self.test_max_step_threshold = 150
        self.test_min_return_threshold = 150


class EnvParams(Mujoco):
    def __init__(self):
        super().__init__()
        self.level = 'CartPole-v0'
        # self.reward_scaling = 20.


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
