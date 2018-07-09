from agents.nec_agent import NECAgentParameters
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from base_parameters import VisualizationParameters
from core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from environments.environment import SingleLevelSelection, SelectedPhaseOnlyDumpMethod, MaxDumpMethod
from environments.gym_environment import Atari, AtariInputFilter, atari_deterministic_v4

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(2000)

#########
# Agent #
#########
agent_params = NECAgentParameters()

agent_params.network_wrappers['main'].learning_rate = 0.00001
agent_params.input_filter = AtariInputFilter()
agent_params.input_filter.remove_reward_filter('clipping')

###############
# Environment #
###############
env_params = Atari()
env_params.level = SingleLevelSelection(atari_deterministic_v4)
env_params.random_initialization_steps = 1

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = False

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
