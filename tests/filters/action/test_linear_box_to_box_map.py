import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from filters.action.linear_box_to_box_map import LinearBoxToBoxMap
from spaces import BoxActionSpace, DiscreteActionSpace
import numpy as np


@pytest.mark.unit_test
def test_filter():
    filter = LinearBoxToBoxMap(10, 20)

    # passing an output space that is wrong
    with pytest.raises(ValueError):
        filter.validate_output_action_space(DiscreteActionSpace(10))

    # 1 dimensional box
    output_space = BoxActionSpace(1, 5, 35)
    input_space = filter.get_unfiltered_action_space(output_space)

    action = np.array([2])
    with pytest.raises(ValueError):
        filter.validate_input_action(action)

    action = np.array([12])
    filter.validate_input_action(action)
    result = filter.filter(action)
    assert result == np.array([11])
    assert output_space.val_matches_space_definition(result)

