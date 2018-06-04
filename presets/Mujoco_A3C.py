from agents.actor_critic_agent import ActorCriticAgentParameters
from block_factories.basic_rl_factory import BasicRLFactory
from block_scheduler import BlockSchedulerParameters
from configurations import VisualizationParameters
from core_types import TrainingSteps, Episodes, EnvironmentSteps, RunPhase
from environments.gym_environment import Mujoco, mujoco_v1, MujocoInputFilter
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod, SingleLevelSelection
from exploration_policies.continuous_entropy import ContinuousEntropyParameters
from filters.observation.observation_normalization_filter import ObservationNormalizationFilter
from filters.reward.reward_rescale_filter import RewardRescaleFilter

####################
# Block Scheduling #
####################
schedule_params = BlockSchedulerParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = Episodes(10000000000)
schedule_params.evaluation_steps = Episodes(1)
schedule_params.heatup_steps = EnvironmentSteps(0)

################
# Agent Params #
################
agent_params = ActorCriticAgentParameters()
agent_params.algorithm.apply_gradients_every_x_episodes = 1
agent_params.algorithm.num_steps_between_gradient_updates = 20
agent_params.algorithm.beta_entropy = 0.005
agent_params.network_wrappers['main'].learning_rate = 0.00002

agent_params.input_filter = MujocoInputFilter()
agent_params.input_filter.add_reward_filter('rescale', RewardRescaleFilter(20))
agent_params.input_filter.add_observation_filter('observation', 'normalize', ObservationNormalizationFilter())

agent_params.exploration = ContinuousEntropyParameters()

###############
# Environment #
###############
env_params = Mujoco()
env_params.level = SingleLevelSelection(mujoco_v1)

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = True

factory = BasicRLFactory(agent_params=agent_params, env_params=env_params,
                         schedule_params=schedule_params, vis_params=vis_params)


