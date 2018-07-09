from agents.n_step_q_agent import NStepQAgentParameters
from architectures.tensorflow_components.architecture import Conv2d, Dense
from base_parameters import VisualizationParameters
from core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from environments.environment import SingleLevelSelection, SelectedPhaseOnlyDumpMethod, MaxDumpMethod
from environments.gym_environment import Atari, atari_deterministic_v4
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10000000)
schedule_params.evaluation_steps = EnvironmentEpisodes(0)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
agent_params = NStepQAgentParameters()

agent_params.network_wrappers['main'].learning_rate = 0.0001
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = [Conv2d([16, 8, 4]),
                                                                                          Conv2d([32, 4, 2])]
agent_params.network_wrappers['main'].middleware_parameters.scheme = [Dense([256])]

###############
# Environment #
###############
env_params = Atari()
env_params.level = SingleLevelSelection(atari_deterministic_v4)

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = False


graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
