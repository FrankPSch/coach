#
# Copyright (c) 2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import gym
import numpy as np

from utils import lower_under_to_upper

try:
    import roboschool
    from OpenGL import GL
except ImportError:
    from logger import failed_imports
    failed_imports.append("RoboSchool")

try:
    from gym_extensions.continuous import mujoco
except:
    from logger import failed_imports
    failed_imports.append("GymExtensions")

try:
    import pybullet_envs
except ImportError:
    from logger import failed_imports
    failed_imports.append("PyBullet")

from typing import Dict, Any, Union
from core_types import RunPhase
from environments.environment import Environment, EnvironmentParameters, LevelSelection
from spaces import Discrete, Box, ObservationSpace, ImageObservationSpace, MeasurementsObservationSpace, StateSpace
from filters.filter import NoInputFilter, NoOutputFilter
from filters.reward.reward_clipping_filter import RewardClippingFilter
from filters.observation.observation_rescale_to_size_filter import ObservationRescaleToSizeFilter
from filters.observation.observation_stacking_filter import ObservationStackingFilter
from filters.observation.observation_rgb_to_y_filter import ObservationRGBToYFilter
from filters.observation.observation_to_uint8_filter import ObservationToUInt8Filter
from filters.filter import InputFilter
import random
from configurations import VisualizationParameters
from collections import OrderedDict


# Parameters

class GymEnvironmentParameters(EnvironmentParameters):
    def __init__(self):
        super().__init__()
        self.random_initialization_steps = 0
        self.max_over_num_frames = 1
        self.additional_simulator_parameters = None

    @property
    def path(self):
        return 'environments.gym_environment:GymEnvironment'


"""
Roboschool Environment Components
"""
RoboSchoolInputFilters = NoInputFilter()
RoboSchoolOutputFilters = NoOutputFilter()


class Roboschool(GymEnvironmentParameters):
    def __init__(self):
        super().__init__()
        self.frame_skip = 1
        self.default_input_filter = RoboSchoolInputFilters
        self.default_output_filter = RoboSchoolOutputFilters


gym_roboschool_envs = ['inverted_pendulum', 'inverted_pendulum_swingup', 'inverted_double_pendulum', 'reacher',
                       'hopper', 'walker2d', 'half_cheetah', 'ant', 'humanoid', 'humanoid_flagrun',
                       'humanoid_flagrun_harder', 'pong']
roboschool_v0 = {e: "{}".format(lower_under_to_upper(e) + '-v0') for e in gym_roboschool_envs}

"""
Mujoco Environment Components
"""
MujocoInputFilter = NoInputFilter()
MujocoOutputFilter = NoOutputFilter()


class Mujoco(GymEnvironmentParameters):
    def __init__(self):
        super().__init__()
        self.frame_skip = 1
        self.default_input_filter = MujocoInputFilter
        self.default_output_filter = MujocoOutputFilter


gym_mujoco_envs = ['inverted_pendulum', 'inverted_double_pendulum', 'reacher', 'hopper', 'walker2d', 'half_cheetah',
                   'ant', 'swimmer', 'humanoid', 'humanoid_standup']
mujoco_v1 = {e: "{}".format(lower_under_to_upper(e) + '-v1') for e in gym_mujoco_envs}
mujoco_v1['walker2d'] = 'Walker2d-v1'


"""
Bullet Environment Components
"""
BulletInputFilter = NoInputFilter()
BulletOutputFilter = NoOutputFilter()


class Bullet(GymEnvironmentParameters):
    def __init__(self):
        super().__init__()
        self.frame_skip = 1
        self.default_input_filter = BulletInputFilter
        self.default_output_filter = BulletOutputFilter


"""
Atari Environment Components
"""

AtariInputFilter = InputFilter(is_a_reference_filter=True)
AtariInputFilter.add_reward_filter('clipping', RewardClippingFilter(-1.0, 1.0))
AtariInputFilter.add_observation_filter('observation', 'rescaling',
                                        ObservationRescaleToSizeFilter(ImageObservationSpace(np.array([84, 84, 3]),
                                                                                             high=255)))
AtariInputFilter.add_observation_filter('observation', 'to_grayscale', ObservationRGBToYFilter())
AtariInputFilter.add_observation_filter('observation', 'to_uint8', ObservationToUInt8Filter(0, 255))
AtariInputFilter.add_observation_filter('observation', 'stacking', ObservationStackingFilter(4))
AtariOutputFilter = NoOutputFilter()


class Atari(GymEnvironmentParameters):
    def __init__(self):
        super().__init__()
        self.frame_skip = 4
        self.max_over_num_frames = 2
        self.random_initialization_steps = 30
        self.default_input_filter = AtariInputFilter
        self.default_output_filter = AtariOutputFilter


gym_atari_envs = ['air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
                  'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
                  'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
                  'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
                  'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
                  'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
                  'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
                  'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
                  'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']
atari_deterministic_v4 = {e: "{}".format(lower_under_to_upper(e) + 'Deterministic-v4') for e in gym_atari_envs}


class MaxOverFramesAndFrameskipEnvWrapper(gym.Wrapper):
    def __init__(self, env, frameskip=4, max_over_num_frames=2):
        super().__init__(env)
        self.max_over_num_frames = max_over_num_frames
        self.observations_stack = []
        self.frameskip = frameskip
        self.first_frame_to_max_over = self.frameskip - self.max_over_num_frames

    def reset(self):
        return self.env.reset()

    def step(self, action):
        total_reward = 0.0
        done = None
        info = None
        self.observations_stack = []
        for i in range(self.frameskip):
            observation, reward, done, info = self.env.step(action)
            if i >= self.first_frame_to_max_over:
                self.observations_stack.append(observation)
            total_reward += reward
            if done:
                # deal with last state in episode
                if not self.observations_stack:
                    self.observations_stack.append(observation)
                break

        max_over_frames_observation = np.max(self.observations_stack, axis=0)

        return max_over_frames_observation, total_reward, done, info


# Environment
class GymEnvironment(Environment):
    def __init__(self, level: LevelSelection, frame_skip: int, visualization_parameters: VisualizationParameters,
                 additional_simulator_parameters: Dict[str, Any] = None, seed: Union[None, int]=None,
                 human_control: bool=False, custom_reward_threshold: Union[int, float]=None,
                 random_initialization_steps: int=1, max_over_num_frames: int=1, **kwargs):
        super().__init__(level, seed, frame_skip, human_control, custom_reward_threshold,
                         visualization_parameters)

        self.random_initialization_steps = random_initialization_steps
        self.max_over_num_frames = max_over_num_frames
        self.additional_simulator_parameters = additional_simulator_parameters

        # load and initialize environment
        if ':' in self.env_id:
            # load custom env
            if self.additional_simulator_parameters:
                self.env = gym.envs.registration.load(self.env_id)(**self.additional_simulator_parameters)
            else:
                self.env = gym.envs.registration.load(self.env_id)()
        else:
            self.env = gym.make(self.env_id)

        # for classic control we want to use the native renderer because otherwise we will get 2 renderer windows
        self.native_rendering = self.native_rendering or \
                                any([env in str(self.env.unwrapped.__class__) for env in ['classic_control']])
        if self.native_rendering:
            self.renderer.close()

        # seed
        if self.seed is not None:
            self.env.seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

        # frame skip and max between consecutive frames
        self.is_atari_env = 'Atari' in self.env.unwrapped.__str__()
        self.timelimit_env_wrapper = self.env
        if self.is_atari_env:
            self.env.unwrapped.frameskip = 1  # this accesses the atari env that is wrapped with a timelimit wrapper env
            self.env = MaxOverFramesAndFrameskipEnvWrapper(self.env,
                                                           frameskip=self.frame_skip,
                                                           max_over_num_frames=self.max_over_num_frames)
        else:
            self.env.unwrapped.frameskip = self.frame_skip

        self.state_space = StateSpace({})

        # observations
        if not isinstance(self.env.observation_space, gym.spaces.dict_space.Dict):
            state_space = {'observation': self.env.observation_space}
        else:
            state_space = self.env.observation_space.spaces

        for observation_space_name, observation_space in state_space.items():
            if len(observation_space.shape) == 3 and observation_space.shape[-1] == 3:
                # we assume gym has image observations which are RGB and where their values are within 0-255
                self.state_space[observation_space_name] = ImageObservationSpace(
                    shape=np.array(observation_space.shape),
                    high=255,
                    channels_axis=-1
                )
            else:
                self.state_space[observation_space_name] = MeasurementsObservationSpace(
                    shape=observation_space.shape[0],
                    low=observation_space.low,
                    high=observation_space.high
                )
        if 'goal' in state_space.keys():
            self.goal_space = self.state_space['goal']

        # actions
        if type(self.env.action_space) == gym.spaces.box.Box:
            self.action_space = Box(
                shape=self.env.action_space.shape,
                low=self.env.action_space.low,
                high=self.env.action_space.high
            )
        elif type(self.env.action_space) == gym.spaces.discrete.Discrete:
            actions_description = []
            if hasattr(self.env.unwrapped, 'get_action_meanings'):
                actions_description = self.env.unwrapped.get_action_meanings()
            self.action_space = Discrete(
                num_actions=self.env.action_space.n,
                descriptions=actions_description
            )

        if self.human_control:
            # TODO: add this to the action space
            # map keyboard keys to actions
            self.key_to_action = {}
            if hasattr(self.env.unwrapped, 'get_keys_to_action'):
                self.key_to_action = self.env.unwrapped.get_keys_to_action()

        # initialize the state by getting a new state from the environment
        self.reset(True)

        # render
        if self.is_rendered:
            image = self.get_rendered_image()
            scale = 1
            if self.human_control:
                scale = 2
            if not self.native_rendering:
                self.renderer.create_screen(image.shape[1]*scale, image.shape[0]*scale)

        # measurements
        if self.env.spec is not None:
            self.timestep_limit = self.env.spec.timestep_limit
        else:
            self.timestep_limit = None

        # the info is only updated after the first step
        self.state = self.step(self.action_space.default_action).new_state
        self.state_space['measurements'] = MeasurementsObservationSpace(shape=len(self.info.keys()))

        if self.env.spec:
            if custom_reward_threshold is None:
                self.reward_success_threshold = self.env.spec.reward_threshold

    def _wrap_state(self, state):
        if not isinstance(self.env.observation_space, gym.spaces.Dict):
            # TODO: add measurements and goal
            return {'observation': state}
        return state

    def _update_state(self):
        if self.is_atari_env and hasattr(self, 'current_ale_lives') \
                and self.current_ale_lives != self.env.unwrapped.ale.lives():
            if self.phase == RunPhase.TRAIN or self.phase == RunPhase.HEATUP:
                # signal termination for life loss
                self.done = True
            elif self.phase == RunPhase.TEST and not self.done:
                # the episode is not terminated in evaluation, but we need to press fire again
                self._press_fire()
            self._update_ale_lives()
        # TODO: update the measurements
        if self.state and "goal" in self.state.keys():
            self.goal = self.state['goal']

    def _take_action(self, action):
        if type(self.action_space) == Box:
            action = self.action_space.clip_action_to_space(action)

        self.state, self.reward, self.done, self.info = self.env.step(action)
        self.state = self._wrap_state(self.state)

    def _random_noop(self):
        # simulate a random initial environment state by stepping for a random number of times between 0 and 30
        step_count = 0
        random_initialization_steps = random.randint(0, self.random_initialization_steps)
        while self.action_space is not None and (self.state is None or step_count < random_initialization_steps):
            step_count += 1
            self.step(self.action_space.default_action)

    def _press_fire(self):
        fire_action = 1
        if self.is_atari_env and self.env.unwrapped.get_action_meanings()[fire_action] == 'FIRE':
            self.current_ale_lives = self.env.unwrapped.ale.lives()
            self.step(fire_action)
            if self.done:
                self.reset()

    def _update_ale_lives(self):
        if self.is_atari_env:
            self.current_ale_lives = self.env.unwrapped.ale.lives()

    def _restart_environment_episode(self, force_environment_reset=False):
        # prevent reset of environment if there are ale lives left
        if (self.is_atari_env and self.env.unwrapped.ale.lives() > 0) \
                and not force_environment_reset and not self.timelimit_env_wrapper._past_limit():
            self.step(self.action_space.default_action)
        else:
            self.state = self.env.reset()
            self.state = self._wrap_state(self.state)
            self._update_ale_lives()
        if self.is_atari_env:
            self._random_noop()
            self._press_fire()

        # initialize the number of lives
        self._update_ale_lives()

    def _render(self):
        self.env.render(mode='human')

    def get_rendered_image(self):
        return self.env.render(mode='rgb_array')
