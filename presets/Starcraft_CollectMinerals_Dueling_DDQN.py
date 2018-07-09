from collections import OrderedDict

from agents.ddqn_agent import DDQNAgentParameters
from architectures.tensorflow_components.heads.dueling_q_head import DuelingQHeadParameters
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from base_parameters import VisualizationParameters
from core_types import RunPhase
from core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod
from environments.starcraft2_environment import StarCraft2EnvironmentParameters
from filters.action.box_discretization import BoxDiscretization
from filters.filter import OutputFilter
from memories.memory import MemoryGranularity
from schedules import LinearSchedule

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(50)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(50000)

#########
# Agent #
#########
agent_params = DDQNAgentParameters()

agent_params.network_wrappers['main'].learning_rate = 0.0001
agent_params.network_wrappers['main'].input_embedders_parameters["screen"].input_rescaling['image'] = 3.
agent_params.network_wrappers['main'].heads_parameters = [DuelingQHeadParameters()]
agent_params.memory.max_size = (MemoryGranularity.Transitions, 1000000)
# slave_agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(10000)
agent_params.exploration.epsilon_schedule = LinearSchedule(1.0, 0.1, 1000000)
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(4)
agent_params.output_filter = \
    OutputFilter(
        action_filters=OrderedDict([
            ('discretization', BoxDiscretization(num_bins_per_dimension=4, force_int_bins=True))
        ]),
        is_a_reference_filter=False
    )


###############
# Environment #
###############

env_params = StarCraft2EnvironmentParameters()
env_params.level = 'CollectMineralShards'


vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = False
# vis_params.dump_in_episode_signals = True

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
