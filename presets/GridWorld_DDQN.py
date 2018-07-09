from collections import OrderedDict

from agents.ddqn_agent import DDQNAgentParameters
from architectures.tensorflow_components.architecture import Dense
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from base_parameters import VisualizationParameters, EmbedderScheme
from core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from environments.environment import SelectedPhaseOnlyDumpMethod, MaxDumpMethod
from environments.gym_environment import Mujoco
from filters.action.partial_discrete_action_space_map import PartialDiscreteActionSpaceMap
from filters.filter import NoInputFilter, OutputFilter
from filters.reward.reward_rescale_filter import RewardRescaleFilter
from memories.memory import MemoryGranularity
from schedules import LinearSchedule

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
agent_params.memory.max_size = (MemoryGranularity.Episodes, 2000)
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(500)
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)
# agent_params.memory.discount = 0.99
agent_params.algorithm.discount = 0.99
agent_params.network_wrappers['main'].learning_rate = 0.00025
agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False
agent_params.exploration.epsilon_schedule = LinearSchedule(1.0, 0.1, 500000)
agent_params.exploration.evaluation_epsilon = 0
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = EmbedderScheme.Shallow
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].input_rescaling['image'] = 7
agent_params.network_wrappers['main'].middleware_parameters.scheme = [Dense([64])]
# agent_params.input_filter.reward_filters['scale'] = RewardRescaleFilter(1/0.001)
agent_params.input_filter = NoInputFilter()
# agent_params.input_filter.observation_filters['rescale'] =\
#     ObservationRescaleToSizeFilter(ObservationSpace(np.array([84, 84, 3])))
agent_params.output_filter = OutputFilter(
        action_filters=OrderedDict([
            ("masking_actions", PartialDiscreteActionSpaceMap(target_actions=[0, 1, 2]))
        ]),
        is_a_reference_filter=False
    )

###############
# Environment #
###############
env_params = Mujoco()
env_params.level = 'gym_minigrid.envs:MultiRoomEnvN2S4'
env_params.level = 'gym_minigrid.envs:MultiRoomEnvN6'
env_params.level = 'gym_minigrid.envs:EmptyEnv6x6'
env_params.level = 'gym_minigrid.envs:MultiRoomEnvN3S9'
env_params.level = 'gym_minigrid.envs:MultiRoomEnvN5S9'
# env_params.seed = 1
env_params.additional_simulator_parameters = {'gridSize': 25}

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = False

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
