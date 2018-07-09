from agents.ddpg_agent import DDPGAgentParameters
from agents.dqn_agent import DQNAgentParameters
from architectures.tensorflow_components.architecture import Dense
from environments.environment import SelectedPhaseOnlyDumpMethod, MaxDumpMethod, SingleLevelSelection
from exploration_policies.e_greedy import EGreedyParameters
from filters.observation.observation_normalization_filter import ObservationNormalizationFilter
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from base_parameters import VisualizationParameters, EmbedderScheme, InputEmbedderParameters
from core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from environments.gym_environment import Mujoco, MujocoInputFilter, fetch_v1
from memories.hindsight_experience_replay import HindsightExperienceReplayParameters, HindsightGoalSelectionMethod
from memories.memory import MemoryGranularity
from schedules import ConstantSchedule
from spaces import GoalsActionSpace, ReachingGoal


####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentEpisodes(16 * 50 * 200)  # 200 epochs
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(16 * 50)  # 50 cycles
schedule_params.evaluation_steps = EnvironmentEpisodes(10)
schedule_params.heatup_steps = EnvironmentSteps(0)

################
# Agent Params #
################
agent_params = DDPGAgentParameters()
agent_params.network_wrappers['actor'].learning_rate = 0.001
agent_params.network_wrappers['critic'].learning_rate = 0.001
agent_params.network_wrappers['actor'].batch_size = 128
agent_params.network_wrappers['critic'].batch_size = 128
agent_params.network_wrappers['actor'].middleware_parameters.scheme = [Dense([64]), Dense([64]), Dense([64])]
agent_params.network_wrappers['critic'].middleware_parameters.scheme = [Dense([64]), Dense([64]), Dense([64])]

# default TF params for Adam
agent_params.network_wrappers['actor'].optimizer_epsilon = 1e-08
agent_params.network_wrappers['actor'].adam_optimizer_beta1 = 0.9
agent_params.network_wrappers['actor'].adam_optimizer_beta1 = 0.999
agent_params.network_wrappers['critic'].optimizer_epsilon = 1e-08
agent_params.network_wrappers['critic'].adam_optimizer_beta1 = 0.9
agent_params.network_wrappers['critic'].adam_optimizer_beta1 = 0.999

# TODO make sure that indeed no input embedder is needed and only concat is done
agent_params.network_wrappers['actor'].input_embedders_parameters = {
    'observation': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
    'desired_goal': InputEmbedderParameters(scheme=EmbedderScheme.Empty)}

agent_params.network_wrappers['critic'].input_embedders_parameters = {
    'action': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
    'observation': InputEmbedderParameters(scheme=EmbedderScheme.Empty)}

agent_params.algorithm.use_target_network_for_evaluation = True
agent_params.algorithm.discount = 0.98
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentEpisodes(16)
agent_params.algorithm.num_consecutive_training_steps = 40
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(40)
agent_params.algorithm.rate_for_copying_weights_to_target = 0.05

agent_params.memory = HindsightExperienceReplayParameters()
agent_params.memory.max_size = (MemoryGranularity.Transitions, 10**6)
agent_params.memory.hindsight_goal_selection_method = HindsightGoalSelectionMethod.Future
agent_params.memory.hindsight_transitions_per_regular_transition = 4
agent_params.memory.goals_action_space = GoalsActionSpace(goal_type='achieved_goal',
                                                          reward_type=ReachingGoal(distance_from_goal_threshold=0.05,
                                                                                   goal_reaching_reward=0,
                                                                                   default_reward=-1),
                                                          distance_metric=GoalsActionSpace.DistanceMetric.Euclidean)
agent_params.exploration = EGreedyParameters()
agent_params.exploration.epsilon_schedule = ConstantSchedule(0.2)
agent_params.exploration.evaluation_epsilon = 0
agent_params.exploration.continuous_exploration_policy_parameters.noise_percentage_schedule = ConstantSchedule(0.05)
# TODO: when acting greedily, add 5% noise of the allowed values on each coordinate
agent_params.input_filter = MujocoInputFilter()
agent_params.input_filter.add_observation_filter('observation', 'normalize', ObservationNormalizationFilter())

###############
# Environment #
###############
env_params = Mujoco()
env_params.level = SingleLevelSelection(fetch_v1)
# env_params.check_successes = True

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = True

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)


# self.algorithm.add_intrinsic_reward_for_reaching_the_goal = False

