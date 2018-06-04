import copy

from agents.actor_critic_agent import ActorCriticAgentParameters
from agents.nec_agent import NECAgentParameters
from block_factories.basic_rl_factory import BasicRLFactory
from block_factories.custom_actor_critic_factory import CustomActorCriticFactory
from configurations import MiddlewareTypes, VisualizationParameters
from environments.gym_environment import MujocoInputFilter, Mujoco
from schedules import LinearSchedule
from filters.reward.reward_rescale_filter import RewardRescaleFilter
from exploration_policies.categorical import CategoricalParameters
from agents.policy_optimization_agent import PolicyGradientRescaler
from block_scheduler import BlockSchedulerParameters
from core_types import TrainingSteps, Episodes, EnvironmentSteps
from memories.memory import MemoryGranularity

"""
This is a reference preset to follow. It should be maintained working and passing tests.  
"""

####################
# Block Scheduling #
####################
schedule_params = BlockSchedulerParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = Episodes(50)
schedule_params.evaluation_steps = Episodes(3)
schedule_params.heatup_steps = EnvironmentSteps(1000)

############################
# ActorCritic Agent Params #
############################
agent_params = ActorCriticAgentParameters()

agent_params.algorithm.policy_gradient_rescaler = PolicyGradientRescaler.GAE
agent_params.algorithm.discount = 0.99
agent_params.algorithm.apply_gradients_every_x_episodes = 1
agent_params.algorithm.num_steps_between_gradient_updates = 5
agent_params.algorithm.gae_lambda = 1
agent_params.algorithm.beta_entropy = 0.01

agent_params.network_wrappers['main'].optimizer_type = 'Adam'
agent_params.network_wrappers['main'].learning_rate = 0.0001
agent_params.network_wrappers['main'].middleware_type = MiddlewareTypes.FC

agent_params.input_filter = MujocoInputFilter()
agent_params.input_filter.add_reward_filter('rescale', RewardRescaleFilter(200))

agent_params.exploration = CategoricalParameters()

######### NEC agent params ######################
nec_agent_params = NECAgentParameters()
nec_agent_params.memory.max_size = (MemoryGranularity.Episodes, 200)
nec_agent_params.exploration.epsilon_schedule = LinearSchedule(0.1, 0.1, 1000)
nec_agent_params.network_wrappers['main'].learning_rate = 0.00025
nec_agent_params.exploration.evaluation_epsilon = 0
nec_agent_params.algorithm.discount = 0.99
nec_agent_params.input_filter = agent_params.input_filter
nec_agent_params.algorithm.heatup_using_network_decisions = False
# nec_agent_params.algorithm.dnd_size = 1000



#################
# CartPole-v0   #
#################
env_params = Mujoco()
env_params.level = 'CartPole-v0'


# set a distributed ER
# nec_agent_params.memory.distributed_memory = True

# factory = CustomActorCriticFactory(actor_params=agent_params, critic_params=nec_agent_params,
#                                    env_params=env_params, vis_params=VisualizationParameters(),
#                                    schedule_params=schedule_params)
# agent_params.algorithm.policy_gradient_rescaler = PolicyGradientRescaler.CUSTOM_ACTOR_CRITIC

factory = BasicRLFactory(agent_params=agent_params, env_params=env_params,
                         schedule_params=schedule_params, vis_params=VisualizationParameters())


def get_custom_actor_critic_factory():
    return CustomActorCriticFactory(actor_params=agent_params, critic_params=nec_agent_params,
                                    env_params=env_params, vis_params=VisualizationParameters(),
                                    schedule_params=schedule_params)
