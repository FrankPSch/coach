from agents.dqn_agent import DQNAgentParameters
from block_factories.basic_rl_factory import BasicRLFactory
from block_scheduler import BlockSchedulerParameters
from configurations import VisualizationParameters, EmbedderScheme, InputEmbedderParameters
from core_types import TrainingSteps, Episodes, EnvironmentSteps, RunPhase
from environments.gym_environment import Atari, atari_deterministic_v4, Mujoco
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod, SingleLevelSelection
from memories.memory import MemoryGranularity
from schedules import LinearSchedule, ConstantSchedule

bit_length = 8

####################
# Block Scheduling #
####################
schedule_params = BlockSchedulerParameters()
schedule_params.improve_steps = TrainingSteps(400000)
schedule_params.steps_between_evaluation_periods = Episodes(16*50)  # 50 cycles
schedule_params.evaluation_steps = Episodes(10)
schedule_params.heatup_steps = EnvironmentSteps(0)

################
# Agent Params #
################
agent_params = DQNAgentParameters()
agent_params.network_wrappers['main'].learning_rate = 0.001
agent_params.network_wrappers['main'].batch_size = 128
agent_params.network_wrappers['main'].middleware_hidden_layer_size = 256
empty_embedder = InputEmbedderParameters()
empty_embedder.embedder_scheme = EmbedderScheme.Empty
agent_params.network_wrappers['main'].input_types = {'state': empty_embedder,
                                                     'goal': empty_embedder}
agent_params.algorithm.discount = 0.98
agent_params.algorithm.num_consecutive_playing_steps = Episodes(16)
agent_params.algorithm.num_consecutive_training_steps = 40
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(40)
agent_params.algorithm.rate_for_copying_weights_to_target = 0.05
agent_params.memory.max_size = (MemoryGranularity.Transitions, 10**6)
agent_params.exploration.epsilon_schedule = ConstantSchedule(0.2)
agent_params.exploration.evaluation_epsilon = 0

###############
# Environment #
###############
env_params = Mujoco()
env_params.level = 'gym_bit_flip:BitFlip'
env_params.additional_simulator_parameters = {'bit_length': bit_length, 'mean_zero': True}
# env_params.custom_reward_threshold = -bit_length + 1
# env_params.check_successes = True

vis_params = VisualizationParameters()

factory = BasicRLFactory(agent_params=agent_params, env_params=env_params,
                         schedule_params=schedule_params, vis_params=vis_params)


# self.algorithm.add_intrinsic_reward_for_reaching_the_goal = False

