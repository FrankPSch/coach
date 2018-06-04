import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from filters.action.box_masking import BoxMasking
from spaces import Box, Discrete
import numpy as np


@pytest.mark.unit_test
def test_filter():
    filter = BoxMasking(10, 20)

    # passing an output space that is wrong
    with pytest.raises(ValueError):
        filter.validate_output_action_space(Discrete(10))

    # 1 dimensional box
    output_space = Box(1, 5, 30)
    input_space = filter.get_unfiltered_action_space(output_space)

    action = np.array([2])
    filter.validate_input_action(action)
    result = filter.filter(action)
    assert result == np.array([12])
    assert output_space.val_matches_space_definition(result)

