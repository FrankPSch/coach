from agents.categorical_dqn_agent import CategoricalDQNAgentParameters
from block_factories.basic_rl_factory import BasicRLFactory
from block_scheduler import BlockSchedulerParameters
from configurations import VisualizationParameters
from core_types import Episodes, EnvironmentSteps, RunPhase
from environments.gym_environment import Atari, atari_deterministic_v4
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod, SingleLevelSelection

####################
# Block Scheduling #
####################
schedule_params = BlockSchedulerParameters()
schedule_params.improve_steps = EnvironmentSteps(20000000)
schedule_params.steps_between_evaluation_periods = Episodes(20)
schedule_params.evaluation_steps = Episodes(1)
schedule_params.heatup_steps = EnvironmentSteps(50000)

################
# Agent Params #
################
agent_params = CategoricalDQNAgentParameters()
agent_params.network_wrappers['main'].learning_rate = 0.00025

###############
# Environment #
###############
env_params = Atari()
env_params.level = SingleLevelSelection(atari_deterministic_v4)

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = True

factory = BasicRLFactory(agent_params=agent_params, env_params=env_params,
                         schedule_params=schedule_params, vis_params=vis_params)
