from agents.bc_agent import BCAgentParameters
from agents.ddqn_agent import DDQNAgentParameters
from block_factories.basic_rl_factory import BasicRLFactory
from block_scheduler import BlockSchedulerParameters
from configurations import VisualizationParameters, OutputTypes
from core_types import TrainingSteps, Episodes, EnvironmentSteps, RunPhase
from environments.gym_environment import AtariInputFilter, Atari
from schedules import LinearSchedule
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod
from filters.observation.observation_crop_filter import ObservationCropFilter
import numpy as np

####################
# Block Scheduling #
####################
from memories.memory import MemoryGranularity

schedule_params = BlockSchedulerParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = TrainingSteps(500)
schedule_params.evaluation_steps = Episodes(5)
schedule_params.heatup_steps = EnvironmentSteps(0)

####################
# BC Agent Params #
####################
agent_params = BCAgentParameters()
agent_params.network_wrappers['main'].learning_rate = 0.00025
agent_params.memory.max_size = (MemoryGranularity.Transitions, 1000000)
agent_params.memory.discount = 0.99
agent_params.algorithm.discount = 0.99
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(0)
agent_params.memory.load_memory_from_file_path = 'datasets/montezuma_revenge.p'

##################################
# Atari PongDeterministic-v4 #
##################################
env_params = Atari()
env_params.level = 'MontezumaRevenge-v0'
env_params.random_initialization_steps = 30

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = True

factory = BasicRLFactory(agent_params=agent_params, env_params=env_params,
                         schedule_params=schedule_params, vis_params=vis_params)
