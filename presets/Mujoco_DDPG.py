from agents.ddpg_agent import DDPGAgentParameters
from block_factories.basic_rl_factory import BasicRLFactory
from block_scheduler import BlockSchedulerParameters
from configurations import VisualizationParameters, EmbedderScheme
from core_types import TrainingSteps, Episodes, EnvironmentSteps, RunPhase
from environments.gym_environment import Mujoco, mujoco_v1, MujocoInputFilter
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod, SingleLevelSelection
from filters.reward.reward_rescale_filter import RewardRescaleFilter

####################
# Block Scheduling #
####################

schedule_params = BlockSchedulerParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = Episodes(20)
schedule_params.evaluation_steps = Episodes(1)
schedule_params.heatup_steps = EnvironmentSteps(1000)

################
# Agent Params #
################
agent_params = DDPGAgentParameters()
agent_params.network_wrappers['actor'].input_types['observation'].embedder_scheme = [[400]]
agent_params.network_wrappers['actor'].middleware_hidden_layer_size = 300
agent_params.network_wrappers['critic'].input_types['observation'].embedder_scheme = [[400]]
agent_params.network_wrappers['critic'].middleware_hidden_layer_size = 300
agent_params.network_wrappers['critic'].input_types['action'].embedder_scheme = EmbedderScheme.Empty
agent_params.input_filter = MujocoInputFilter()
agent_params.input_filter.add_reward_filter("rescale", RewardRescaleFilter(10))

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
