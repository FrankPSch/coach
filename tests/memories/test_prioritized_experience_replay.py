# nasty hack to deal with issue #46
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import numpy as np

from core_types import Transition
from memories.prioritized_experience_replay import SegmentTree


@pytest.mark.unit_test
def test_sum_tree():
    # test power of 2 sum tree
    sum_tree = SegmentTree(size=4, operation=SegmentTree.Operation.SUM)
    sum_tree.add(10, "10")
    assert sum_tree.total_value() == 10
    sum_tree.add(20, "20")
    assert sum_tree.total_value() == 30
    sum_tree.add(5, "5")
    assert sum_tree.total_value() == 35
    sum_tree.add(7.5, "7.5")
    assert sum_tree.total_value() == 42.5
    sum_tree.add(2.5, "2.5")
    assert sum_tree.total_value() == 35
    sum_tree.add(5, "5")
    assert sum_tree.total_value() == 20

    assert sum_tree.get(10) == (5, 5.0, '5')

    sum_tree.update(5, 10)
    assert sum_tree.__str__() == "[25.]\n[ 7.5 17.5]\n[ 2.5  5.  10.   7.5]\n"

    # test non power of 2 sum tree
    with pytest.raises(ValueError):
        sum_tree = SegmentTree(size=5, operation=SegmentTree.Operation.SUM)


@pytest.mark.unit_test
def test_min_tree():
    min_tree = SegmentTree(size=4, operation=SegmentTree.Operation.MIN)
    min_tree.add(10, "10")
    assert min_tree.total_value() == 10
    min_tree.add(20, "20")
    assert min_tree.total_value() == 10
    min_tree.add(5, "5")
    assert min_tree.total_value() == 5
    min_tree.add(7.5, "7.5")
    assert min_tree.total_value() == 5
    min_tree.add(2, "2")
    assert min_tree.total_value() == 2
    min_tree.add(3, "3")
    min_tree.add(3, "3")
    min_tree.add(3, "3")
    min_tree.add(1, "1")
    assert min_tree.total_value() == 1
