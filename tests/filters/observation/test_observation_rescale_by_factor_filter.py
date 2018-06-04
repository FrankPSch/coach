import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import numpy as np

from filters.observation.observation_rescale_by_factor_filter import ObservationRescaleByFactorFilter, RescaleInterpolationType
from spaces import ObservationSpace
from core_types import EnvResponse
from collections import OrderedDict
from filters.filter import InputFilter

@pytest.mark.filterwarnings('ignore:Conversion of')
@pytest.mark.unit_test
def test_filter():
    # make an RGB observation smaller
    env_response = EnvResponse(new_state={'observation': np.ones([20, 30, 3])}, reward=0, game_over=False)
    rescale_filter = InputFilter()
    rescale_filter.add_observation_filter('observation', 'rescale',
                                         ObservationRescaleByFactorFilter(0.5, RescaleInterpolationType.BILINEAR))

    result = rescale_filter.filter(env_response)
    unfiltered_observation = env_response.new_state['observation']
    filtered_observation = result.new_state['observation']

    # make sure the original observation is unchanged
    assert unfiltered_observation.shape == (20, 30, 3)

    # validate the shape of the filtered observation
    assert filtered_observation.shape == (10, 15, 3)

    # make a grayscale observation bigger
    env_response = EnvResponse(new_state={'observation': np.ones([20, 30])}, reward=0, game_over=False)
    rescale_filter = InputFilter()
    rescale_filter.add_observation_filter('observation', 'rescale',
                                         ObservationRescaleByFactorFilter(2, RescaleInterpolationType.BILINEAR))
    result = rescale_filter.filter(env_response)
    filtered_observation = result.new_state['observation']

    # validate the shape of the filtered observation
    assert filtered_observation.shape == (40, 60)
    assert np.all(filtered_observation == np.ones([40, 60]))


@pytest.mark.unit_test
def test_get_filtered_observation_space():
    # error on wrong number of channels
    rescale_filter = InputFilter()
    rescale_filter.add_observation_filter('observation', 'rescale',
                                         ObservationRescaleByFactorFilter(0.5, RescaleInterpolationType.BILINEAR))
    observation_space = ObservationSpace(np.array([10, 20, 5]))
    with pytest.raises(ValueError):
        filtered_observation_space = rescale_filter.get_filtered_observation_space('observation', observation_space)

    # error on wrong number of dimensions
    observation_space = ObservationSpace(np.array([10, 20, 10, 3]))
    with pytest.raises(ValueError):
        filtered_observation_space = rescale_filter.get_filtered_observation_space('observation', observation_space)

    # make sure the new observation space shape is calculated correctly
    observation_space = ObservationSpace(np.array([10, 20, 3]))
    filtered_observation_space = rescale_filter.get_filtered_observation_space('observation', observation_space)
    assert np.all(filtered_observation_space.shape == np.array([5, 10, 3]))

    # make sure the original observation space is unchanged
    assert np.all(observation_space.shape == np.array([10, 20, 3]))
