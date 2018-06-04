from agents.dqn_agent import DQNAgentParameters
from block_factories.basic_rl_factory import BasicRLFactory
from block_scheduler import BlockSchedulerParameters
from configurations import VisualizationParameters
from core_types import TrainingSteps, Episodes, EnvironmentSteps
from environments.gym_environment import MujocoInputFilter, Mujoco
from schedules import LinearSchedule

####################
# Block Scheduling #
####################
from memories.memory import MemoryGranularity

schedule_params = BlockSchedulerParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = Episodes(100)
schedule_params.evaluation_steps = Episodes(1)
schedule_params.heatup_steps = EnvironmentSteps(1000)


####################
# DQN Agent Params #
####################
agent_params = DQNAgentParameters()
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(100)
agent_params.network_wrappers['main'].learning_rate = 0.00025
agent_params.memory.max_size = (MemoryGranularity.Transitions, 40000)
agent_params.exploration.epsilon_schedule = LinearSchedule(0.5, 0.01, 3000)
agent_params.algorithm.discount = 1.0
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)

#TODO-fixme
# agent_params.test = True
# agent_params.test_max_step_threshold = 150
# agent_params.test_min_return_threshold = 150

##############################
#      Gym CartPole-v0       #
##############################
env_params = Mujoco()
env_params.level = 'CartPole-v0'

factory = BasicRLFactory(agent_params=agent_params, env_params=env_params,
                         schedule_params=schedule_params, vis_params=VisualizationParameters())
