import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import numpy as np

from filters.observation.observation_stacking_filter import ObservationStackingFilter
from spaces import ObservationSpace
from core_types import EnvResponse
from filters.filter import InputFilter
from collections import OrderedDict


@pytest.fixture
def env_response():
    observation = np.random.rand(20, 30, 1)
    return EnvResponse(new_state={'observation': observation}, reward=0, game_over=False)


@pytest.fixture
def stack_filter():
    stack_filter = InputFilter()
    stack_filter.add_observation_filter('observation', 'stack', ObservationStackingFilter(4, stacking_axis=-1))
    return stack_filter


@pytest.mark.unit_test
def test_filter(stack_filter, env_response):
    # stack observation on empty stack
    result = stack_filter.filter(env_response)
    unfiltered_observation = env_response.new_state['observation']
    filtered_observation = result.new_state['observation']

    # validate that the shape of the unfiltered observation is unchanged
    assert unfiltered_observation.shape == (20, 30, 1)
    assert np.array(filtered_observation).shape == (20, 30, 1, 4)
    assert np.all(np.array(filtered_observation)[:, :, :, -1] == unfiltered_observation)

    # stack observation on non-empty stack
    result = stack_filter.filter(env_response)
    filtered_observation = result.new_state['observation']
    assert np.array(filtered_observation).shape == (20, 30, 1, 4)


@pytest.mark.unit_test
def test_get_filtered_observation_space(stack_filter, env_response):
    observation_space = ObservationSpace(np.array([5, 10, 20]))
    filtered_observation_space = stack_filter.get_filtered_observation_space('observation', observation_space)

    # make sure the new observation space shape is calculated correctly
    assert np.all(filtered_observation_space.shape == np.array([5, 10, 20, 4]))

    # make sure the original observation space is unchanged
    assert np.all(observation_space.shape == np.array([5, 10, 20]))

    # call after stack is already created with non-matching shape -> error
    result = stack_filter.filter(env_response)
    with pytest.raises(ValueError):
        filtered_observation_space = stack_filter.get_filtered_observation_space('observation', observation_space)


@pytest.mark.unit_test
def test_reset(stack_filter, env_response):
    # stack observation on empty stack
    result = stack_filter.filter(env_response)
    unfiltered_observation = env_response.new_state['observation']
    filtered_observation = result.new_state['observation']

    assert np.all(np.array(filtered_observation)[:, :, :, -1] == unfiltered_observation)

    # reset and make sure the outputs are correct
    stack_filter.reset()
    unfiltered_observation = np.random.rand(20, 30, 1)
    new_env_response = EnvResponse(new_state={'observation': unfiltered_observation}, reward=0, game_over=False)
    result = stack_filter.filter(new_env_response)
    filtered_observation = result.new_state['observation']
    assert np.all(np.array(filtered_observation)[:, :, :, 0] == unfiltered_observation)
    assert np.all(np.array(filtered_observation)[:, :, :, -1] == unfiltered_observation)
