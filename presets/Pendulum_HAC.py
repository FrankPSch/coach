import numpy as np

from agents.ddpg_agent import DDPGAgentParameters
from agents.hac_ddpg_agent import HACDDPGAgentParameters
from architectures.tensorflow_components.architecture import Dense
from graph_managers.hrl_graph_manager import HRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from base_parameters import VisualizationParameters, EmbeddingMergerType, EmbedderScheme, InputEmbedderParameters
from core_types import EnvironmentEpisodes, EnvironmentSteps, RunPhase, TrainingSteps
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod
from environments.gym_environment import Mujoco
from exploration_policies.e_greedy import EGreedyParameters
from exploration_policies.ou_process import OUProcessParameters
from memories.hindsight_experience_replay import HindsightExperienceReplayParameters, HindsightGoalSelectionMethod
from memories.memory import MemoryGranularity
from schedules import ConstantSchedule
from spaces import GoalsActionSpace, ReachingGoal

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentEpisodes(500000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(4 * 64)  # 4 small batches of 64 episodes
schedule_params.evaluation_steps = EnvironmentEpisodes(64)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########

time_limit = 1000
polar_coordinates = False
if polar_coordinates:
    distance_from_goal_threshold = np.array([0.075, 0.75])
else:
    distance_from_goal_threshold = np.array([0.075, 0.075, 0.75])
goals_action_space = GoalsActionSpace('observation',
                                      ReachingGoal(default_reward=-1,
                                                   goal_reaching_reward=0,
                                                   distance_from_goal_threshold=distance_from_goal_threshold),
                                      lambda goal, state: np.abs(goal - state))  # raw L1 distance

# top agent
top_agent_params = DDPGAgentParameters()

top_agent_params.memory = HindsightExperienceReplayParameters()
top_agent_params.memory.max_size = (MemoryGranularity.Transitions, 10000000)
top_agent_params.memory.hindsight_transitions_per_regular_transition = 3
top_agent_params.memory.hindsight_goal_selection_method = HindsightGoalSelectionMethod.Future
top_agent_params.memory.goals_action_space = goals_action_space
top_agent_params.memory.action_replay = True
top_agent_params.algorithm.num_consecutive_playing_steps = EnvironmentEpisodes(32)
top_agent_params.algorithm.num_consecutive_training_steps = 40
top_agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(40)

# exploration - OU process
top_agent_params.exploration = OUProcessParameters()
top_agent_params.exploration.theta = 0.1

# actor
top_actor = top_agent_params.network_wrappers['actor']
top_actor.input_embedders_parameters = {'observation': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
                                        'desired_goal': InputEmbedderParameters(scheme=EmbedderScheme.Empty)}
top_actor.middleware_parameters.scheme = [Dense([64])] * 3
top_actor.learning_rate = 0.001
top_actor.batch_size = 4096

# critic
top_critic = top_agent_params.network_wrappers['critic']
top_critic.input_embedders_parameters = {'observation': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
                                         'action': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
                                         'desired_goal': InputEmbedderParameters(scheme=EmbedderScheme.Empty)}
top_critic.embedding_merger_type = EmbeddingMergerType.Concat
top_critic.middleware_parameters.scheme = [Dense([64])] * 3
top_critic.learning_rate = 0.001
top_critic.batch_size = 4096

# ----------

# bottom agent
bottom_agent_params = HACDDPGAgentParameters()
bottom_agent_params.algorithm.in_action_space = goals_action_space
bottom_agent_params.memory = HindsightExperienceReplayParameters()
bottom_agent_params.memory.max_size = (MemoryGranularity.Transitions, 12000000)
bottom_agent_params.memory.hindsight_transitions_per_regular_transition = 4
bottom_agent_params.memory.hindsight_goal_selection_method = HindsightGoalSelectionMethod.Future
bottom_agent_params.memory.goals_action_space = goals_action_space
bottom_agent_params.algorithm.num_consecutive_playing_steps = EnvironmentEpisodes(16 * 25)  # 25 episodes is one true env episode
bottom_agent_params.algorithm.num_consecutive_training_steps = 40
bottom_agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(40)

bottom_agent_params.exploration = EGreedyParameters()
bottom_agent_params.exploration.epsilon_schedule = ConstantSchedule(0.2)
bottom_agent_params.exploration.evaluation_epsilon = 0
bottom_agent_params.exploration.continuous_exploration_policy_parameters = OUProcessParameters()
bottom_agent_params.exploration.continuous_exploration_policy_parameters.theta = 0.1

# actor
bottom_actor = bottom_agent_params.network_wrappers['actor']
bottom_actor.input_embedders_parameters = {'observation': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
                                           'desired_goal': InputEmbedderParameters(scheme=EmbedderScheme.Empty)}
bottom_actor.middleware_parameters.scheme = [Dense([64])] * 3
bottom_actor.learning_rate = 0.001
bottom_actor.batch_size = 4096

# critic
bottom_critic = bottom_agent_params.network_wrappers['critic']
bottom_critic.input_embedders_parameters = {'observation': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
                                            'action': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
                                            'desired_goal': InputEmbedderParameters(scheme=EmbedderScheme.Empty)}
bottom_critic.embedding_merger_type = EmbeddingMergerType.Concat
bottom_critic.middleware_parameters.scheme = [Dense([64])] * 3
bottom_critic.learning_rate = 0.001
bottom_critic.batch_size = 4096

agents_params = [top_agent_params, bottom_agent_params]

###############
# Environment #
###############
env_params = Mujoco()
env_params.level = "environments.mujoco.pendulum_with_goals:PendulumWithGoals"
env_params.additional_simulator_parameters = {"time_limit": time_limit,
                                              "random_goals_instead_of_standing_goal": False,
                                              "polar_coordinates": polar_coordinates,
                                              "goal_reaching_thresholds": distance_from_goal_threshold}
env_params.frame_skip = 10
env_params.custom_reward_threshold = -time_limit + 1

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST)]
vis_params.dump_mp4 = False
vis_params.native_rendering = False

graph_manager = HRLGraphManager(agents_params=agents_params, env_params=env_params,
                                schedule_params=schedule_params, vis_params=vis_params,
                                consecutive_steps_to_run_each_level=EnvironmentSteps(40))
