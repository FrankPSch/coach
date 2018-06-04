import numpy as np

from agents.nec_agent import NECAgentParameters
from block_factories.basic_rl_factory import BasicRLFactory
from configurations import VisualizationParameters
from environments.gym_environment import Atari, AtariInputFilter, MujocoInputFilter
from filters.filter import InputFilter
from schedules import ConstantSchedule, LinearSchedule
from filters.observation.observation_crop_filter import ObservationCropFilter
from block_scheduler import BlockSchedulerParameters
from core_types import TrainingSteps, Episodes, EnvironmentSteps
from filters.reward.reward_rescale_filter import RewardRescaleFilter
from memories.memory import MemoryGranularity

####################
# Block Scheduling #
####################
schedule_params = BlockSchedulerParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = Episodes(5)
schedule_params.evaluation_steps = Episodes(1)
schedule_params.heatup_steps = EnvironmentSteps(1300)

####################
# NEC Agent Params #
####################
agent_params = NECAgentParameters()

agent_params.network_wrappers['main'].learning_rate = 0.00025
agent_params.exploration.epsilon_schedule = LinearSchedule(0.5, 0.1, 1000)
agent_params.exploration.evaluation_epsilon = 0
agent_params.algorithm.discount = 0.99
agent_params.memory.max_size = (MemoryGranularity.Episodes, 200)
agent_params.input_filter = MujocoInputFilter()
agent_params.input_filter.add_reward_filter('rescale', RewardRescaleFilter(200))


##############################
# Atari PongDeterministic-v4 #
##############################
env_params = Atari()
env_params.level = 'CartPole-v0'

factory = BasicRLFactory(agent_params=agent_params, env_params=env_params,
                         schedule_params=schedule_params, vis_params=VisualizationParameters())
