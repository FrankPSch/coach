from agents.ppo_agent import PPOAgentParameters
from block_factories.basic_rl_factory import BasicRLFactory
from block_scheduler import BlockSchedulerParameters
from configurations import VisualizationParameters
from core_types import TrainingSteps, Episodes, EnvironmentSteps, RunPhase
from environments.gym_environment import Mujoco, mujoco_v1, MujocoInputFilter
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod, SingleLevelSelection
from exploration_policies.continuous_entropy import ContinuousEntropyParameters
from filters.observation.observation_normalization_filter import ObservationNormalizationFilter

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
agent_params = PPOAgentParameters()
agent_params.network_wrappers['actor'].learning_rate = 0.001
agent_params.network_wrappers['critic'].learning_rate = 0.001

agent_params.network_wrappers['actor'].input_types['observation'].embedder_scheme = [[64]]
agent_params.network_wrappers['actor'].middleware_hidden_layer_size = 64
agent_params.network_wrappers['critic'].input_types['observation'].embedder_scheme = [[64]]
agent_params.network_wrappers['critic'].middleware_hidden_layer_size = 64

agent_params.input_filter = MujocoInputFilter()
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


