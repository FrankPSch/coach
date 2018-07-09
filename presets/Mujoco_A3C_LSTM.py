from agents.actor_critic_agent import ActorCriticAgentParameters
from architectures.tensorflow_components.architecture import Dense
from architectures.tensorflow_components.middlewares.lstm_middleware import LSTMMiddlewareParameters
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from base_parameters import VisualizationParameters, InputEmbedderParameters, MiddlewareScheme, TestingParameters
from core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod, SingleLevelSelection
from environments.gym_environment import Mujoco, mujoco_v2, MujocoInputFilter
from exploration_policies.continuous_entropy import ContinuousEntropyParameters
from filters.observation.observation_normalization_filter import ObservationNormalizationFilter
from filters.reward.reward_rescale_filter import RewardRescaleFilter

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10000000000)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
agent_params = ActorCriticAgentParameters()
agent_params.algorithm.apply_gradients_every_x_episodes = 1
agent_params.algorithm.num_steps_between_gradient_updates = 20
agent_params.algorithm.beta_entropy = 0.005
agent_params.network_wrappers['main'].learning_rate = 0.00002
agent_params.network_wrappers['main'].input_embedders_parameters['observation'] = \
    InputEmbedderParameters(scheme=[Dense([200])])
agent_params.network_wrappers['main'].middleware_parameters = LSTMMiddlewareParameters(scheme=MiddlewareScheme.Empty,
                                                                                       number_of_lstm_cells=128)

agent_params.input_filter = MujocoInputFilter()
agent_params.input_filter.add_reward_filter('rescale', RewardRescaleFilter(1/20.))
agent_params.input_filter.add_observation_filter('observation', 'normalize', ObservationNormalizationFilter())

agent_params.exploration = ContinuousEntropyParameters()

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
test_params.min_reward_threshold = 200 / 20.
test_params.max_episodes_to_achieve_reward = 200
test_params.num_workers = 8
test_params.level = 'inverted_pendulum'

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    test_params=test_params)


