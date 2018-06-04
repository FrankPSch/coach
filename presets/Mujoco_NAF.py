from agents.naf_agent import NAFAgentParameters
from block_factories.basic_rl_factory import BasicRLFactory
from block_scheduler import BlockSchedulerParameters
from configurations import VisualizationParameters, EmbedderScheme
from core_types import TrainingSteps, Episodes, EnvironmentSteps, RunPhase
from environments.gym_environment import Mujoco, mujoco_v1
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod, SingleLevelSelection

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
agent_params = NAFAgentParameters()
agent_params.network_wrappers['main'].input_types['observation'].embedder_scheme = [[200]]
agent_params.network_wrappers['main'].middleware_hidden_layer_size = 200

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
