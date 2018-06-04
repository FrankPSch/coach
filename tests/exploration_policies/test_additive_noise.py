import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest

from spaces import Discrete, Box
from exploration_policies.additive_noise import AdditiveNoise
from schedules import LinearSchedule
import numpy as np
from core_types import RunPhase


@pytest.mark.unit_test
def test_init():
    # discrete control
    action_space = Discrete(3)
    noise_schedule = LinearSchedule(1.0, 1.0, 1000)

    # additive noise doesn't work for discrete controls
    with pytest.raises(ValueError):
        policy = AdditiveNoise(action_space, noise_schedule, 0)

    # additive noise requires a bounded range for the actions
    action_space = Box(np.array([10]))
    with pytest.raises(ValueError):
        policy = AdditiveNoise(action_space, noise_schedule, 0)


@pytest.mark.unit_test
def test_get_action():
    # make sure noise is in range
    action_space = Box(np.array([10]), -1, 1)
    noise_schedule = LinearSchedule(1.0, 1.0, 1000)
    policy = AdditiveNoise(action_space, noise_schedule, 0)

    # the action range is 2, so there is a ~0.1% chance that the noise will be larger than 3*std=3*2=6
    for i in range(1000):
        action = policy.get_action(np.zeros([10]))
        assert np.all(action < 10)
