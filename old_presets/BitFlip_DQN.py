from configurations import *
from block_factories.basic_rl_factory import BasicRLFactory


class AgentParams(AgentParameters):
    def __init__(self):
        AgentParameters.__init__(self, DQN, ExplorationParameters, None)
        # bit_length = np.random.randint(8, 60)
        bit_length = 8
        pass
        pass

        pass
        pass

        self.learning_rate = 0.001
        self.batch_size = 128
        self.evaluate_every_x_training_iterations = 40 * 50
        self.evaluation_episodes = 100
        self.algorithm.num_training_iterations = 400000
        self.num_training_iterations = 400000
        self.num_heatup_steps = 0

        self.algorithm.optimizer_type = 'Adam'
        self.algorithm.discount = 0.98
        self.algorithm.num_consecutive_playing_episodes = 16
        self.algorithm.num_consecutive_training_steps = 40
        self.algorithm.num_training_iterations_between_copying_online_weights_to_target = 40
        self.algorithm.num_steps_between_copying_online_weights_to_target = None
        self.algorithm.rate_for_copying_weights_to_target = 0.05
        self.algorithm.middleware_type = MiddlewareTypes.FC
        self.algorithm.hidden_layers_activation_function = 'relu'
        self.algorithm.hidden_layers_size = 256
        self.algorithm.num_episodes_in_experience_replay = None
        self.algorithm.num_transitions_in_experience_replay = 10**6
        empty_input_embedder = InputEmbedderParameters()
        empty_input_embedder.embedder_scheme = EmbedderDepth.Empty
        self.algorithm.input_types = [empty_input_embedder, empty_input_embedder]
        self.algorithm.state_values_to_use = ['observation', 'goal']

        self.algorithm.add_intrinsic_reward_for_reaching_the_goal = False

        self.exploration.epsilon_decay_steps = 1
        self.exploration.initial_epsilon = 0.2
        self.exploration.final_epsilon = 0.2
        self.exploration.evaluation_epsilon = 0.0

        # the learning schedule specified above means that these summaries would happen too frequently
        pass


class EnvParams(Mujoco):
    def __init__(self):
        super().__init__()
        self.level = 'gym_bit_flip:BitFlip'
        self.additional_simulator_parameters = {'bit_length': bit_length, 'mean_zero': True}
        self.custom_reward_threshold = -bit_length + 1
        self.check_successes = True


class VisParams(VisualizationParameters):
    def __init__(self):
        super().__init__()
        self.print_summary = False

factory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)
