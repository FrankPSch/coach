from agents.actor_critic_agent import ActorCriticAgentParameters
from block_factories.basic_rl_factory import BasicRLFactory
from configurations import MiddlewareTypes, VisualizationParameters
from environments.environment import SingleLevelSelection, SelectedPhaseOnlyDumpMethod, MaxDumpMethod
from environments.gym_environment import Atari, atari_deterministic_v4
from exploration_policies.categorical import CategoricalParameters
from block_scheduler import BlockSchedulerParameters
from core_types import TrainingSteps, Episodes, EnvironmentSteps, RunPhase


####################
# Block Scheduling #
####################
schedule_params = BlockSchedulerParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = Episodes(100000000)
schedule_params.evaluation_steps = Episodes(1)
schedule_params.heatup_steps = EnvironmentSteps(10000)

################
# Agent Params #
################
agent_params = ActorCriticAgentParameters()

agent_params.algorithm.apply_gradients_every_x_episodes = 1
agent_params.algorithm.num_steps_between_gradient_updates = 20
agent_params.algorithm.beta_entropy = 0.05

agent_params.network_wrappers['main'].middleware_type = MiddlewareTypes.FC
agent_params.network_wrappers['main'].learning_rate = 0.0001

agent_params.exploration = CategoricalParameters()

###############
# Environment #
###############
env_params = Atari()
env_params.level = SingleLevelSelection(atari_deterministic_v4)

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = True


factory = BasicRLFactory(agent_params=agent_params, env_params=env_params,
                         schedule_params=schedule_params, vis_params=VisualizationParameters())
