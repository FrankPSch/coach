from agents.nec_agent import NECAgentParameters
from block_factories.basic_rl_factory import BasicRLFactory
from configurations import VisualizationParameters
from environments.environment import SingleLevelSelection
from environments.gym_environment import Atari, AtariInputFilter, atari_deterministic_v4
from block_scheduler import BlockSchedulerParameters
from core_types import TrainingSteps, Episodes, EnvironmentSteps

####################
# Block Scheduling #
####################
schedule_params = BlockSchedulerParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = Episodes(20)
schedule_params.evaluation_steps = Episodes(1)
schedule_params.heatup_steps = EnvironmentSteps(2000)

################
# Agent Params #
################
agent_params = NECAgentParameters()

agent_params.network_wrappers['main'].learning_rate = 0.00001
agent_params.input_filter = AtariInputFilter()
agent_params.input_filter.remove_reward_filter('clipping')

###############
# Environment #
###############
env_params = Atari()
env_params.level = SingleLevelSelection(atari_deterministic_v4)
env_params.random_initialization_steps = 1

factory = BasicRLFactory(agent_params=agent_params, env_params=env_params,
                         schedule_params=schedule_params, vis_params=VisualizationParameters())
