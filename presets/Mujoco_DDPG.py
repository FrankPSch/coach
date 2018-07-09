from agents.ddpg_agent import DDPGAgentParameters
from architectures.tensorflow_components.architecture import Dense
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from base_parameters import VisualizationParameters, EmbeddingMergerType, MiddlewareScheme, \
    EmbedderScheme, TestingParameters
from core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase, GradientClippingMethod
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod, SingleLevelSelection
from environments.gym_environment import Mujoco, mujoco_v2, MujocoInputFilter
from filters.reward.reward_rescale_filter import RewardRescaleFilter

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(2000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(1000)

#########
# Agent #
#########
agent_params = DDPGAgentParameters()
agent_params.network_wrappers['actor'].input_embedders_parameters['observation'].scheme = [Dense([400])]
agent_params.network_wrappers['actor'].middleware_parameters.scheme = [Dense([300])]
agent_params.network_wrappers['critic'].input_embedders_parameters['observation'].scheme = [Dense([400])]
agent_params.network_wrappers['critic'].middleware_parameters.scheme = [Dense([300])]
agent_params.network_wrappers['critic'].input_embedders_parameters['action'].scheme = [Dense([400])]
# agent_params.network_wrappers['critic'].embedding_merger_type = EmbeddingMergerType.Sum
# # agent_params.network_wrappers['actor'].gradients_clipping_method = GradientClippingMethod.ClipByValue
# # agent_params.network_wrappers['actor'].clip_gradients = 1
# # agent_params.input_filter = MujocoInputFilter()
# # agent_params.input_filter.add_reward_filter("rescale", RewardRescaleFilter(1/10.))

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
test_params.max_episodes_to_achieve_reward = 650
test_params.level = 'inverted_pendulum'

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    test_params=test_params)
