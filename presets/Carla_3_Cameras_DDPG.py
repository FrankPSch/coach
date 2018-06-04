import copy

from agents.ddpg_agent import DDPGAgentParameters
from block_factories.basic_rl_factory import BasicRLFactory
from block_scheduler import BlockSchedulerParameters
from configurations import VisualizationParameters
from core_types import TrainingSteps, Episodes, EnvironmentSteps, RunPhase
from environments.carla_environment import CarlaEnvironmentParameters, CameraTypes, CarlaInputFilter
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod

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
agent_params = DDPGAgentParameters()
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(4)

# front camera
agent_params.network_wrappers['actor'].input_types['forward_camera'] = \
    agent_params.network_wrappers['actor'].input_types.pop('observation')
agent_params.network_wrappers['critic'].input_types['forward_camera'] = \
    agent_params.network_wrappers['critic'].input_types.pop('observation')

# left camera
agent_params.network_wrappers['actor'].input_types['left_camera'] = \
    copy.deepcopy(agent_params.network_wrappers['actor'].input_types['forward_camera'])
agent_params.network_wrappers['critic'].input_types['left_camera'] = \
    copy.deepcopy(agent_params.network_wrappers['critic'].input_types['forward_camera'])

# right camera
agent_params.network_wrappers['actor'].input_types['right_camera'] = \
    copy.deepcopy(agent_params.network_wrappers['actor'].input_types['forward_camera'])
agent_params.network_wrappers['critic'].input_types['right_camera'] = \
    copy.deepcopy(agent_params.network_wrappers['critic'].input_types['forward_camera'])

agent_params.input_filter = CarlaInputFilter()
agent_params.input_filter.copy_filters_from_one_observation_to_another('forward_camera', 'left_camera')
agent_params.input_filter.copy_filters_from_one_observation_to_another('forward_camera', 'right_camera')

###############
# Environment #
###############
env_params = CarlaEnvironmentParameters()
env_params.level = 'town1'
env_params.cameras = [CameraTypes.FRONT, CameraTypes.LEFT, CameraTypes.RIGHT]

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = True

factory = BasicRLFactory(agent_params=agent_params, env_params=env_params,
                         schedule_params=schedule_params, vis_params=vis_params)
