import os
import sys

from block_factories.block_factory import TaskParameters
from core_types import QActionStateValue

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import tensorflow as tf
from tensorflow import logging
import pytest
logging.set_verbosity(logging.INFO)


@pytest.mark.unit_test
def test_get_QActionStateValue_predictions():
    tf.reset_default_graph()
    from presets.CartPole_DQN import factory as cartpole_dqn_factory
    assert cartpole_dqn_factory
    block_scheduler = cartpole_dqn_factory.create_block(task_parameters=
                                                        TaskParameters(framework_type="tensorflow",
                                                                       experiment_path="./experiments/test"))
    assert block_scheduler
    block_scheduler.improve_steps.num_steps = 1
    block_scheduler.steps_between_evaluation_periods.num_steps = 5

    # block_scheduler.improve()
    #
    # agent = block_scheduler.level_managers[0].composite_agents['simple_rl_agent'].agents['simple_rl_agent/agent']
    # some_state = agent.memory.sample(1)[0].state
    # cartpole_dqn_predictions = agent.get_predictions(states=some_state, prediction_type=QActionStateValue)
    # assert cartpole_dqn_predictions.shape == (1, 2)


if __name__ == '__main__':
    test_get_QActionStateValue_predictions()
