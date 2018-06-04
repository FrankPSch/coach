from agents.ddqn_agent import DDQNAgentParameters
from agents.dqn_agent import DQNAgentParameters
from block_factories.basic_rl_factory import BasicRLFactory
from block_scheduler import BlockSchedulerParameters
from configurations import VisualizationParameters, OutputTypes
from core_types import TrainingSteps, Episodes, EnvironmentSteps
from environments.doom_environment import DoomEnvironmentParameters, DoomInputFilter, DoomOutputFilter
from schedules import LinearSchedule

####################
# Block Scheduling #
####################
from memories.memory import MemoryGranularity

schedule_params = BlockSchedulerParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = Episodes(50)
schedule_params.evaluation_steps = Episodes(3)
schedule_params.heatup_steps = EnvironmentSteps(1000)


####################
# DQN Agent Params #
####################
agent_params = DDQNAgentParameters()
agent_params.memory.max_size = (MemoryGranularity.Episodes, 1000)
agent_params.network_wrappers['main'].learning_rate = 0.00025
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(1000)
agent_params.exploration.epsilon_schedule = LinearSchedule(0, 0, 50000)
agent_params.exploration.evaluation_epsilon = 0
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)
agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False
agent_params.network_wrappers['main'].output_types = [OutputTypes.DuelingQ]

##############################
#      Doom Basic       #
##############################
env_params = DoomEnvironmentParameters()
env_params.level = 'basic'

factory = BasicRLFactory(agent_params=agent_params, env_params=env_params,
                         schedule_params=schedule_params, vis_params=VisualizationParameters())
