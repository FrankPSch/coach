from agents.dqn_agent import DQNAgentParameters
from base_parameters import VisualizationParameters
from core_types import EnvironmentSteps, RunPhase
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod, SingleLevelSelection
from environments.gym_environment import Atari, atari_deterministic_v4
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from memories.prioritized_experience_replay import PrioritizedExperienceReplayParameters
from schedules import LinearSchedule

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(50000000)
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(250000)
schedule_params.evaluation_steps = EnvironmentSteps(135000)
schedule_params.heatup_steps = EnvironmentSteps(50000)

#########
# Agent #
#########
agent_params = DQNAgentParameters()
agent_params.network_wrappers['main'].learning_rate = 0.00025
agent_params.memory = PrioritizedExperienceReplayParameters()
agent_params.memory.beta = LinearSchedule(0.4, 1, 12500000)  # 12.5M training iterations = 50M steps = 200M frames

###############
# Environment #
###############
env_params = Atari()
env_params.level = SingleLevelSelection(atari_deterministic_v4)

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = False

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
