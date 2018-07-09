from agents.qr_dqn_agent import QuantileRegressionDQNAgentParameters
from base_parameters import VisualizationParameters
from core_types import EnvironmentSteps, RunPhase
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod, SingleLevelSelection
from environments.gym_environment import Atari, atari_deterministic_v4
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(50000000)
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(250000)
schedule_params.evaluation_steps = EnvironmentSteps(135000)
schedule_params.heatup_steps = EnvironmentSteps(50000)

#########
# Agent #
#########
agent_params = QuantileRegressionDQNAgentParameters()
agent_params.network_wrappers['main'].learning_rate = 0.00005  # called alpha in the paper
agent_params.algorithm.huber_loss_interval = 1  # k = 0 for strict quantile loss, k = 1 for Huber quantile loss

###############
# Environment #
###############
env_params = Atari()
env_params.level = SingleLevelSelection(atari_deterministic_v4)
env_params.seed = 1

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = False

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
