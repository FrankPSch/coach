# nasty hack to deal with issue #46
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# print(sys.path)

import pytest
import numpy as np

from core_types import Transition, GoalTypes
from memories.memory import Episode, MemoryGranularity
from memories.hindsight_experience_replay import HindsightExperienceReplay, HindsightExperienceReplayParameters, \
    HindsightGoalSelectionMethod


class Parameters(HindsightExperienceReplayParameters):
    def __init__(self):
        super().__init__()
        self.max_size = (MemoryGranularity.Transitions, 100)
        self.hindsight_transitions_per_regular_transition = 4
        self.hindsight_goal_selection_method = HindsightGoalSelectionMethod.Future
        self.goal_type = 'observation'#GoalTypes.Observation
        self.distance_from_goal_threshold = 0.1


@pytest.fixture
def episode():
    episode = []
    for i in range(10):
        episode.append(Transition(
            state={'observation': np.array([i]), 'goal': np.array([i])},
            action=i,
        ))
    return episode


@pytest.fixture
def her():
    params = Parameters().__dict__

    import inspect
    args = set(inspect.getfullargspec(HindsightExperienceReplay.__init__).args).intersection(params)
    params = {k: params[k] for k in args}

    return HindsightExperienceReplay(**params)


@pytest.mark.unit_test
def test_sample_goal(her, episode):
    assert her._sample_goal(episode, 8) == 9


@pytest.mark.unit_test
def test_sample_goal_range(her, episode):
    unseen_goals = set(range(1, 9))
    for _ in range(500):
        unseen_goals -= set([int(her._sample_goal(episode, 0))])
        if not unseen_goals:
            return

    assert unseen_goals == set()


@pytest.mark.unit_test
def test_update_episode(her):
    for i in range(10):
        her.store(Transition(
            state={'observation': np.array([i]), 'goal': np.array([i + 1])},
            goal=np.array([i + 1]),
            action=i,
            game_over=i == 9,
            reward=0 if i == 9 else -1,
        ))
        # print('her._num_transitions', her._num_transitions)

    # 10 original transitions, and 9 transitions * 4 hindsight episodes
    assert her.num_transitions() == 10 + (4 * 9)

    # make sure that the goal state was never sampled from the past
    for transition in her.transitions:
        assert transition.state['goal'] > transition.state['observation']
        assert transition.next_state['goal'] >= transition.next_state['observation']

        if transition.reward == 0:
            assert transition.game_over
        else:
            assert not transition.game_over

# test_update_episode(her())