import math
from agents.ddqn_agent import DDQNAgentParameters
from agents.dqn_agent import DQNAgentParameters
from architectures.tensorflow_components.heads.dueling_q_head import DuelingQHeadParameters
from environments.environment import SelectedPhaseOnlyDumpMethod, MaxDumpMethod
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from base_parameters import VisualizationParameters, TestingParameters
from core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from environments.gym_environment import Mujoco
from memories.memory import MemoryGranularity
from schedules import LinearSchedule

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(1000)

#########
# Agent #
#########
agent_params = DDQNAgentParameters()

# DDQN params
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(100)
agent_params.algorithm.discount = 0.99
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)

# NN configuration
agent_params.network_wrappers['main'].learning_rate = 0.00025
agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False
agent_params.network_wrappers['main'].heads_parameters = [DuelingQHeadParameters()]
agent_params.network_wrappers['main'].rescale_gradient_from_head_by_factor = [1/math.sqrt(2), 1/math.sqrt(2)]

# ER size
agent_params.memory.max_size = (MemoryGranularity.Transitions, 40000)

# E-Greedy schedule
agent_params.exploration.epsilon_schedule = LinearSchedule(1.0, 0.01, 10000)

################
#  Environment #
################
env_params = Mujoco()
env_params.level = 'CartPole-v0'

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = False

########
# Test #
########
test_params = TestingParameters()
test_params.test = True
test_params.min_reward_threshold = 150
test_params.max_episodes_to_achieve_reward = 250

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    test_params=test_params)
