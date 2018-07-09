from agents.actor_critic_agent import ActorCriticAgentParameters
from agents.policy_optimization_agent import PolicyGradientRescaler
from environments.environment import SelectedPhaseOnlyDumpMethod, MaxDumpMethod
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from base_parameters import VisualizationParameters, TestingParameters
from core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from environments.gym_environment import MujocoInputFilter, Mujoco
from exploration_policies.categorical import CategoricalParameters
from filters.reward.reward_rescale_filter import RewardRescaleFilter


####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(50)
schedule_params.evaluation_steps = EnvironmentEpisodes(3)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
agent_params = ActorCriticAgentParameters()

agent_params.algorithm.policy_gradient_rescaler = PolicyGradientRescaler.GAE
agent_params.algorithm.discount = 0.99
agent_params.algorithm.apply_gradients_every_x_episodes = 1
agent_params.algorithm.num_steps_between_gradient_updates = 5
agent_params.algorithm.gae_lambda = 1
agent_params.algorithm.beta_entropy = 0.01

agent_params.network_wrappers['main'].optimizer_type = 'Adam'
agent_params.network_wrappers['main'].learning_rate = 0.0001

agent_params.input_filter = MujocoInputFilter()
agent_params.input_filter.add_reward_filter('rescale', RewardRescaleFilter(1/200.))

agent_params.exploration = CategoricalParameters()

###############
# Environment #
###############
env_params = Mujoco()
env_params.level = 'CartPole-v0'

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = False

########
# Test #
########
test_params = TestingParameters()
test_params.test = True
test_params.min_reward_threshold = 0.75
test_params.max_episodes_to_achieve_reward = 200
test_params.num_workers = 8

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    test_params=test_params)
