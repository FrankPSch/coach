import numpy as np
from agents.actor_critic_agent import ActorCriticAgentParameters
from agents.policy_optimization_agent import PolicyGradientRescaler
from architectures.tensorflow_components.architecture import Conv2d, Dense
from architectures.tensorflow_components.heads.policy_head import PolicyHeadParameters
from architectures.tensorflow_components.heads.v_head import VHeadParameters
from architectures.tensorflow_components.middlewares.fc_middleware import FCMiddlewareParameters
from base_parameters import VisualizationParameters, InputEmbedderParameters
from core_types import RunPhase
from core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod
from environments.starcraft2_environment import StarCraft2EnvironmentParameters, StarcraftInputFilter
from exploration_policies.categorical import CategoricalParameters
from filters.filter import NoOutputFilter
from filters.observation.observation_rescale_to_size_filter import ObservationRescaleToSizeFilter
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from spaces import PlanarMapsObservationSpace

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(100000000)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
agent_params = ActorCriticAgentParameters()

agent_params.algorithm.policy_gradient_rescaler = PolicyGradientRescaler.GAE
agent_params.algorithm.apply_gradients_every_x_episodes = 1
agent_params.algorithm.num_steps_between_gradient_updates = 20
agent_params.algorithm.gae_lambda = 0.99
agent_params.algorithm.beta_entropy = 0.01

# TODO: use one-hot encoding channel wise followed by 1x1 convolution

main_network = agent_params.network_wrappers['main']

main_network.input_embedders_parameters = {
    "screen": InputEmbedderParameters(input_rescaling={'image': 3.0},
                                      scheme=[Conv2d([16, 8, 4]),
                                              Conv2d([32, 4, 2])]),
    "minimap": InputEmbedderParameters(input_rescaling={'image': 3.0},
                                       scheme=[Conv2d([16, 8, 4]),
                                               Conv2d([32, 4, 2])]),
    "measurements": InputEmbedderParameters(scheme=[Dense([32])], activation_function='tanh')
}
main_network.middleware_parameters = FCMiddlewareParameters(scheme=[Dense([256])])

# value head + multiple policy head which are created automatically for each sub action in the compound action
main_network.heads_parameters = [VHeadParameters(), PolicyHeadParameters()]

# TODO: v head should also use the n-step returns

main_network.clip_gradients = 40.0
main_network.learning_rate = 0.0001
agent_params.algorithm.num_steps_between_gradient_updates = 30

agent_params.input_filter = StarcraftInputFilter()  # TODO: we need to rescale measurements with log
agent_params.input_filter['screen']['rescaling'] = ObservationRescaleToSizeFilter(
                                                PlanarMapsObservationSpace(np.array([64, 64, 17]),
                                                                           low=0, high=255, channels_axis=-1))
agent_params.input_filter['minimap']['rescaling'] = ObservationRescaleToSizeFilter(
                                                PlanarMapsObservationSpace(np.array([64, 64, 7]),
                                                                           low=0, high=255, channels_axis=-1))
agent_params.output_filter = NoOutputFilter()

agent_params.exploration = CategoricalParameters()

###############
# Environment #
###############

env_params = StarCraft2EnvironmentParameters()
env_params.level = 'CollectMineralShards'
env_params.screen_size = 64
env_params.use_full_action_space = True

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = False

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
