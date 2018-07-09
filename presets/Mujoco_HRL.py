from agents.ddpg_agent import DDPGAgentParameters
from architectures.tensorflow_components.architecture import Dense
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.hrl_graph_manager import HRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from base_parameters import VisualizationParameters, EmbeddingMergerType, MiddlewareScheme, \
    EmbedderScheme, InputEmbedderParameters, TestingParameters
from core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase, GradientClippingMethod
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod, SingleLevelSelection
from environments.gym_environment import Mujoco, mujoco_v2, MujocoInputFilter
from filters.reward.reward_rescale_filter import RewardRescaleFilter

####################
# Graph Scheduling #
####################
from spaces import GoalsActionSpace, InverseDistanceFromGoal

schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(2000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(100)

#########
# Agent #
#########
top_agent_params = DDPGAgentParameters()
top_agent_params.network_wrappers['actor'].input_embedders_parameters['observation'].scheme = [Dense([400])]
top_agent_params.network_wrappers['actor'].middleware_parameters.scheme = [Dense([300])]
top_agent_params.network_wrappers['critic'].input_embedders_parameters['observation'].scheme = [Dense([400])]
top_agent_params.network_wrappers['critic'].middleware_parameters.scheme = [Dense([300])]
top_agent_params.network_wrappers['critic'].input_embedders_parameters['action'].scheme = [Dense([400])]
top_agent_params.network_wrappers['critic'].embedding_merger_type = EmbeddingMergerType.Sum


bottom_agent_params = DDPGAgentParameters()
bottom_agent_params.algorithm.in_action_space = GoalsActionSpace('observation',
                                                                 InverseDistanceFromGoal(0.1, max_reward=10),
                                                                 GoalsActionSpace.DistanceMetric.Euclidean)
bottom_agent_params.network_wrappers['actor'].input_embedders_parameters['observation'].scheme = [Dense([400])]
bottom_agent_params.network_wrappers['actor'].input_embedders_parameters['goal'] = \
    InputEmbedderParameters(scheme=[Dense([400])], batchnorm=True)
bottom_agent_params.network_wrappers['actor'].embedding_merger_type = EmbeddingMergerType.Concat
bottom_agent_params.network_wrappers['actor'].middleware_parameters.scheme = [Dense([300])]

bottom_agent_params.network_wrappers['critic'].input_embedders_parameters['observation'].scheme = [Dense([400])]
bottom_agent_params.network_wrappers['critic'].input_embedders_parameters['action'].scheme = [Dense([400])]
bottom_agent_params.network_wrappers['critic'].input_embedders_parameters['goal'] = \
    InputEmbedderParameters(scheme=[Dense([400])], batchnorm=True)
bottom_agent_params.network_wrappers['critic'].middleware_parameters.scheme = [Dense([300])]
bottom_agent_params.network_wrappers['critic'].embedding_merger_type = EmbeddingMergerType.Concat

agents_params = [top_agent_params, bottom_agent_params]

###############
# Environment #
###############
env_params = Mujoco()
env_params.level = SingleLevelSelection(mujoco_v2)

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = False

########
# Test #
########
test_params = TestingParameters()
test_params.test = True
test_params.min_reward_threshold = 200
test_params.max_episodes_to_achieve_reward = 200
test_params.level = 'inverted_pendulum'

graph_manager = HRLGraphManager(agents_params=agents_params, env_params=env_params,
                                schedule_params=schedule_params, vis_params=vis_params,
                                consecutive_steps_to_run_each_level=EnvironmentSteps(5),
                                test_params=test_params)
