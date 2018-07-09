from agents.nec_agent import NECAgentParameters
from environments.environment import SelectedPhaseOnlyDumpMethod, MaxDumpMethod
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from base_parameters import VisualizationParameters, TestingParameters
from core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from environments.gym_environment import Atari, MujocoInputFilter
from filters.reward.reward_rescale_filter import RewardRescaleFilter
from memories.memory import MemoryGranularity
from schedules import LinearSchedule

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(5)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(1300)

#########
# Agent #
#########

agent_params = NECAgentParameters()

agent_params.network_wrappers['main'].learning_rate = 0.00025
agent_params.exploration.epsilon_schedule = LinearSchedule(0.5, 0.1, 1000)
agent_params.exploration.evaluation_epsilon = 0
agent_params.algorithm.discount = 0.99
agent_params.memory.max_size = (MemoryGranularity.Episodes, 200)
agent_params.input_filter = MujocoInputFilter()
agent_params.input_filter.add_reward_filter('rescale', RewardRescaleFilter(1/200.))

###############
# Environment #
###############
env_params = Atari()
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
test_params.max_episodes_to_achieve_reward = 150

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    test_params=test_params)
