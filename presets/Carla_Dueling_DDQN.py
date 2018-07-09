import math

from agents.ddqn_agent import DDQNAgentParameters
from architectures.tensorflow_components.heads.dueling_q_head import DuelingQHeadParameters
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from base_parameters import VisualizationParameters, MiddlewareScheme
from core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from environments.carla_environment import CarlaEnvironmentParameters
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod
from filters.action.box_discretization import BoxDiscretization
from filters.filter import OutputFilter

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(1000)

#########
# Agent #
#########
agent_params = DDQNAgentParameters()
agent_params.network_wrappers['main'].learning_rate = 0.00025
agent_params.network_wrappers['main'].heads_parameters = [DuelingQHeadParameters()]
agent_params.network_wrappers['main'].middleware_parameters.scheme = MiddlewareScheme.Empty
agent_params.network_wrappers['main'].rescale_gradient_from_head_by_factor = [1/math.sqrt(2), 1/math.sqrt(2)]
agent_params.network_wrappers['main'].clip_gradients = 10
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(4)
agent_params.network_wrappers['main'].input_embedders_parameters['forward_camera'] = \
    agent_params.network_wrappers['main'].input_embedders_parameters.pop('observation')
agent_params.output_filter = OutputFilter()
agent_params.output_filter.add_action_filter('discretization', BoxDiscretization(5))

###############
# Environment #
###############
env_params = CarlaEnvironmentParameters()
env_params.level = 'town1'

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = False

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
