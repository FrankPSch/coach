import math
from agents.ddqn_agent import DDQNAgentParameters
from architectures.tensorflow_components.heads.dueling_q_head import DuelingQHeadParameters
from base_parameters import VisualizationParameters, MiddlewareScheme
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
agent_params = DDQNAgentParameters()
agent_params.network_wrappers['main'].learning_rate = 0.00025
agent_params.network_wrappers['main'].middleware_parameters.scheme = MiddlewareScheme.Empty
agent_params.network_wrappers['main'].heads_parameters = [DuelingQHeadParameters()]
agent_params.network_wrappers['main'].rescale_gradient_from_head_by_factor = [1/math.sqrt(2), 1/math.sqrt(2)]
agent_params.network_wrappers['main'].clip_gradients = 10

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
