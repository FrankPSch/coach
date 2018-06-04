import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from environments.gym_environment import GymEnvironment
from configurations import VisualizationParameters
import numpy as np
from spaces import Discrete, Box, ImageObservationSpace, MeasurementsObservationSpace


@pytest.fixture()
def atari_env():
    # create a breakout gym environment
    env = GymEnvironment(level='Breakout-v0',
                         seed=1,
                         frame_skip=4,
                         visualization_parameters=VisualizationParameters())
    return env


# TODO: change this to pendulum since mujoco needs license
@pytest.fixture()
def continuous_env():
    # create a breakout gym environment
    env = GymEnvironment(level='Pendulum-v0',
                         seed=1,
                         frame_skip=1,
                         visualization_parameters=VisualizationParameters())
    return env


@pytest.mark.unit_test
def test_gym_discrete_environment(atari_env):
    # observation space
    assert type(atari_env.state_space['observation']) == ImageObservationSpace
    assert np.all(atari_env.state_space['observation'].shape == [210, 160, 3])
    assert np.all(atari_env.last_env_response.new_state['observation'].shape == (210, 160, 3))

    # action space
    assert type(atari_env.action_space) == Discrete
    assert np.all(atari_env.action_space.high == 3)

    # make sure that the seed is working properly
    assert np.sum(atari_env.last_env_response.new_state['observation']) == 4115856


@pytest.mark.unit_test
def test_gym_continuous_environment(continuous_env):
    # observation space
    assert type(continuous_env.state_space['observation']) == MeasurementsObservationSpace
    assert np.all(continuous_env.state_space['observation'].shape == [3])
    assert np.all(continuous_env.last_env_response.new_state['observation'].shape == (3,))

    # action space
    assert type(continuous_env.action_space) == Box
    assert np.all(continuous_env.action_space.shape == np.array([1]))

    # make sure that the seed is working properly
    assert np.sum(continuous_env.last_env_response.new_state['observation']) == 1.2661630859028832


@pytest.mark.unit_test
def test_step(atari_env):
    result = atari_env.step(0)

if __name__ == '__main__':
    test_gym_continuous_environment(continuous_env())