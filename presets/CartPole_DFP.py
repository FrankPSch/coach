from agents.dfp_agent import DFPAgentParameters, HandlingTargetsAfterEpisodeEnd
from environments.environment import SelectedPhaseOnlyDumpMethod, MaxDumpMethod
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from base_parameters import VisualizationParameters, EmbedderScheme, TestingParameters
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
schedule_params.heatup_steps = EnvironmentSteps(100)


#########
# Agent #
#########
agent_params = DFPAgentParameters()
agent_params.network_wrappers['main'].learning_rate = 0.0001
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = EmbedderScheme.Medium
agent_params.network_wrappers['main'].input_embedders_parameters['goal'].scheme = EmbedderScheme.Medium
agent_params.network_wrappers['main'].input_embedders_parameters['measurements'].scheme = EmbedderScheme.Medium
agent_params.exploration.epsilon_schedule = LinearSchedule(0.5, 0.01, 3000)
agent_params.exploration.evaluation_epsilon = 0.01
agent_params.algorithm.discount = 1.0
agent_params.algorithm.use_accumulated_reward_as_measurement = True
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)
agent_params.algorithm.goal_vector = [1]  # accumulated_reward
agent_params.algorithm.handling_targets_after_episode_end = HandlingTargetsAfterEpisodeEnd.LastStep

###############
# Environment #
###############
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
test_params.max_episodes_to_achieve_reward = 100

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    test_params=test_params)
