from agents.ppo_agent import PPOAgentParameters
from architectures.tensorflow_components.architecture import Dense
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from base_parameters import VisualizationParameters, TestingParameters
from core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod, SingleLevelSelection
from environments.gym_environment import Mujoco, mujoco_v2, MujocoInputFilter
from exploration_policies.continuous_entropy import ContinuousEntropyParameters
from filters.observation.observation_normalization_filter import ObservationNormalizationFilter

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
agent_params = PPOAgentParameters()
agent_params.network_wrappers['actor'].learning_rate = 0.001
agent_params.network_wrappers['critic'].learning_rate = 0.001

agent_params.network_wrappers['actor'].input_embedders_parameters['observation'].scheme = [Dense([64])]
agent_params.network_wrappers['actor'].middleware_parameters.scheme = [Dense([64])]
agent_params.network_wrappers['critic'].input_embedders_parameters['observation'].scheme = [Dense([64])]
agent_params.network_wrappers['critic'].middleware_parameters.scheme = [Dense([64])]

agent_params.input_filter = MujocoInputFilter()
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
test_params.min_reward_threshold = 200
test_params.max_episodes_to_achieve_reward = 200
test_params.level = 'inverted_pendulum'

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    test_params=test_params)


