import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import tensorflow as tf
from block_factories.block_factory import TaskParameters, DistributedTaskParameters
from utils import get_open_port
from multiprocessing import Process
from tensorflow import logging
import pytest
logging.set_verbosity(logging.INFO)


@pytest.mark.unit_test
def test_basic_rl_factory_with_pong_a3c():
    tf.reset_default_graph()
    from presets.Atari_A3C import factory
    assert factory
    factory.env_params.level = "PongDeterministic-v4"
    block_scheduler = factory.create_block(task_parameters=TaskParameters(framework_type="tensorflow",
                                                                          experiment_path="./experiments/test"))
    assert block_scheduler
    # block_scheduler.improve()


@pytest.mark.unit_test
def test_basic_rl_factory_with_ant_a3c():
    tf.reset_default_graph()
    from presets.Mujoco_A3C import factory
    assert factory
    factory.env_params.level = "Ant-v1"
    block_scheduler = factory.create_block(task_parameters=TaskParameters(framework_type="tensorflow",
                                                                          experiment_path="./experiments/test"))
    assert block_scheduler
    # block_scheduler.improve()


@pytest.mark.unit_test
def test_basic_rl_factory_with_pong_nec():
    tf.reset_default_graph()
    from presets.Atari_NEC import factory
    assert factory
    factory.env_params.level = "PongDeterministic-v4"
    block_scheduler = factory.create_block(task_parameters=TaskParameters(framework_type="tensorflow",
                                                                          experiment_path="./experiments/test"))
    assert block_scheduler
    # block_scheduler.improve()


@pytest.mark.unit_test
def test_basic_rl_factory_with_cartpole_dqn():
    tf.reset_default_graph()
    from presets.CartPole_DQN import factory
    assert factory
    block_scheduler = factory.create_block(task_parameters=TaskParameters(framework_type="tensorflow",
                                                                          experiment_path="./experiments/test"))
    assert block_scheduler
    # block_scheduler.improve()


@pytest.mark.unit_test
def test_basic_rl_factory_with_doom_basic_dqn():
    tf.reset_default_graph()
    from presets.Doom_Basic_DQN import factory
    assert factory
    block_scheduler = factory.create_block(task_parameters=TaskParameters(framework_type="tensorflow",
                                                                          experiment_path="./experiments/test"))
    assert block_scheduler
    # block_scheduler.improve()


# def test_basic_rl_factory_multithreaded_with_pong_a3c():
#     tf.reset_default_graph()
#     from block_scheduler import start_block
#     from presets.Pong_A3C import factory
#     assert factory
#     num_threads = 4
#     ps_hosts = "localhost:{}".format(get_open_port())
#     worker_hosts = ",".join(["localhost:{}".format(get_open_port()) for i in range(num_threads)])
#
#     task_parameters = DistributedTaskParameters(framework_type="tensorflow",
#                                                 parameters_server_hosts=ps_hosts,
#                                                 worker_hosts=worker_hosts,
#                                                 job_type="ps",
#                                                 task_index=0)
#     p = Process(target=start_block, args=("presets.Pong_A3C:factory", task_parameters))
#     p.start()
#
#     for i in range(num_threads):
#         task_parameters = DistributedTaskParameters(framework_type="tensorflow",
#                                                     parameters_server_hosts=ps_hosts,
#                                                     worker_hosts=worker_hosts,
#                                                     job_type="worker",
#                                                     task_index=i)
#         p = Process(target=start_block, args=("presets.Pong_A3C:factory", task_parameters))
#         p.start()
#

if __name__ == '__main__':
    pass
    # test_basic_rl_factory_with_pong_a3c()
    # test_basic_rl_factory_with_ant_a3c()
    # test_basic_rl_factory_with_pong_nec()
	# test_basic_rl_factory_with_cartpole_dqn()
    #test_basic_rl_factory_multithreaded_with_pong_a3c()
	#test_basic_rl_factory_with_doom_basic_dqn()