#
# Copyright (c) 2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from block_scheduler import BlockScheduler
from configurations import Frameworks, Parameters, unfold_dict_or_list, iterable_to_items
from logger import screen
from utils import set_cpu
from collections import OrderedDict


class TaskParameters(Parameters):
    def __init__(self, framework_type: str, evaluate_only: bool=False, use_cpu: bool=False, experiment_path=None,
                 seed=None):
        """
        :param framework_type: deep learning framework type. can be either tensorflow or neon
        :param evaluate_only: the task will be used only for evaluating the model
        :param use_cpu: use the cpu for this task
        :param experiment_path: the path to the directory which will store all the experiment outputs
        :param seed: a seed to use for the random numbers generator
        """
        self.framework_type = framework_type
        self.task_index = None  # TODO: not really needed
        self.evaluate_only = evaluate_only
        self.use_cpu = use_cpu
        self.experiment_path = experiment_path
        self.seed = seed


class DistributedTaskParameters(TaskParameters):
    def __init__(self, framework_type: str, parameters_server_hosts: str, worker_hosts: str, job_type: str,
                 task_index: int, evaluate_only: bool=False, num_tasks: int=None,
                 num_training_tasks: int=None, use_cpu: bool=False, experiment_path=None, dnd=None,
                 shared_memory_scratchpad=None, seed=None):
        """
        :param framework_type: deep learning framework type. can be either tensorflow or neon
        :param evaluate_only: the task will be used only for evaluating the model
        :param parameters_server_hosts: comma-separated list of hostname:port pairs to which the parameter servers are
                                        assigned
        :param worker_hosts: comma-separated list of hostname:port pairs to which the workers are assigned
        :param job_type: the job type - either ps (short for parameters server) or worker
        :param task_index: the index of the process
        :param num_tasks: the number of total tasks that are running (not including the parameters server)
        :param num_training_tasks: the number of tasks that are training (not including the parameters server)
        :param use_cpu: use the cpu for this task
        :param experiment_path: the path to the directory which will store all the experiment outputs
        :param dnd: an external DND to use for NEC. This is a workaround needed for a shared DND not using the scratchpad.
        :param seed: a seed to use for the random numbers generator
        """
        super().__init__(framework_type=framework_type, evaluate_only=evaluate_only, use_cpu=use_cpu,
                         experiment_path=experiment_path, seed=seed)
        self.parameters_server_hosts = parameters_server_hosts
        self.worker_hosts = worker_hosts
        self.job_type = job_type
        self.task_index = task_index
        self.num_tasks = num_tasks
        self.num_training_tasks = num_training_tasks
        self.device = None  # the replicated device which will be used for the global parameters
        self.worker_target = None
        self.dnd = dnd
        self.shared_memory_scratchpad = shared_memory_scratchpad


class BlockFactory(object):
    """
    A block factory is responsible for creating and initializing a block, including all its internal components.
    """
    def __init__(self):  # TODO: configure framework from outside
        self.sess = None

    def create_block(self, task_parameters: TaskParameters) -> BlockScheduler:
        if isinstance(task_parameters, DistributedTaskParameters):
            screen.log_title("Creating block - name: {} task id: {} type: {}".format(self.__class__.__name__,
                                                                                     task_parameters.task_index,
                                                                                     task_parameters.job_type))
        else:
            screen.log_title("Creating block - name: {}".format(self.__class__.__name__))

        # "hide" the gpu if necessary
        if task_parameters.use_cpu:
            set_cpu()

        # create a target server for the worker and a device
        if isinstance(task_parameters, DistributedTaskParameters):
            task_parameters.worker_target, task_parameters.device = \
                self.create_worker_or_parameters_server(task_parameters=task_parameters)

        # create the block modules (and all the graph ops)
        block_scheduler = self._create_block(task_parameters)

        # create a session (it needs to be created after all the graph ops were created)
        # TODO: set the checkpoint dir and secs
        self.sess = self.create_session(task_parameters=task_parameters, checkpoint_dir=None, save_checkpoint_secs=None)

        # set the session for all the modules
        block_scheduler.set_session(self.sess)

        return block_scheduler

    def _create_block(self, task_parameters: TaskParameters) -> BlockScheduler:
        """
        Create all the block modules and the block scheduler
        :param task_parameters: the parameters of the task
        :return: the initialized block scheduler
        """
        raise NotImplementedError("")

    # def set_framework(self, framework_type: Frameworks):
    #     # choosing neural network framework
    #     framework = Frameworks().get(framework_type)
    #     sess = None
    #     if framework == Frameworks.TensorFlow:
    #         import tensorflow as tf
    #         config = tf.ConfigProto()
    #         config.allow_soft_placement = True
    #         config.gpu_options.allow_growth = True
    #         config.gpu_options.per_process_gpu_memory_fraction = 0.2
    #         sess = tf.Session(config=config)
    #     elif framework == Frameworks.Neon:
    #         import ngraph as ng
    #         sess = ng.transformers.make_transformer()
    #     screen.log_title("Using {} framework".format(Frameworks().to_string(framework)))
    #     return sess

    def create_worker_or_parameters_server(self, task_parameters: DistributedTaskParameters):
        from architectures.tensorflow_components.distributed_tf_utils import create_and_start_parameters_server, \
            create_cluster_spec, create_worker_server_and_device

        # create cluster spec
        cluster_spec = create_cluster_spec(parameters_server=task_parameters.parameters_server_hosts,
                                           workers=task_parameters.worker_hosts)

        # create and start parameters server (non-returning function) or create a worker and a device setter
        if task_parameters.job_type == "ps":
            create_and_start_parameters_server(cluster_spec=cluster_spec)
        elif task_parameters.job_type == "worker":
            return create_worker_server_and_device(cluster_spec=cluster_spec,
                                                   task_index=task_parameters.task_index,
                                                   use_cpu=task_parameters.use_cpu)
        else:
            raise ValueError("The job type should be either ps or worker and not {}"
                             .format(task_parameters.job_type))

    def create_session(self, task_parameters: DistributedTaskParameters, checkpoint_dir: str=None,
                       save_checkpoint_secs: int=None):
        import tensorflow as tf
        config = tf.ConfigProto()
        config.allow_soft_placement = True  # allow placing ops on cpu if they are not fit for gpu
        config.gpu_options.allow_growth = True  # allow the gpu memory allocated for the worker to grow if needed

        if isinstance(task_parameters, DistributedTaskParameters):
            # the distributed tensorflow setting
            from architectures.tensorflow_components.distributed_tf_utils import create_monitored_session
            sess = create_monitored_session(target=task_parameters.worker_target,
                                            task_index=task_parameters.task_index,
                                            checkpoint_dir=checkpoint_dir,
                                            save_checkpoint_secs=save_checkpoint_secs,
                                            config=config)
        else:
            # regular session
            sess = tf.Session(config=config)

        return sess

    def __str__(self):
        result = ""
        for key, val in self.__dict__.items():
            params = ""
            if isinstance(val, list) or isinstance(val, dict) or isinstance(val, OrderedDict):
                items = iterable_to_items(val)
                for k, v in items:
                    params += "{}: {}\n".format(k, v)
            else:
                params = val
            result += "{}: \n{}\n".format(key, params)

        return result
