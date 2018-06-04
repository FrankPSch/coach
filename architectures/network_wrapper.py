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

from collections import OrderedDict
from configurations import Frameworks, AgentParameters
from logger import failed_imports, screen
import os
from spaces import ActionSpace, ObservationSpace, MeasurementsObservationSpace, SpacesDefinition
from block_factories.block_factory import DistributedTaskParameters

try:
    import tensorflow as tf
    from architectures.tensorflow_components.general_network import GeneralTensorFlowNetwork
except ImportError:
    failed_imports.append("TensorFlow")

try:
    from architectures.neon_components.general_network import GeneralNeonNetwork
except ImportError:
    failed_imports.append("Neon")


class NetworkWrapper(object):
    """
    Contains multiple networks and managers syncing and gradient updates
    between them.
    """
    def __init__(self, agent_parameters: AgentParameters, has_target: bool, has_global: bool, name: str,
                 spaces: SpacesDefinition, replicated_device=None, worker_device=None):
        self.ap = agent_parameters
        self.network_parameters = self.ap.network_wrappers[name]
        self.has_target = has_target
        self.has_global = has_global
        self.name = name
        self.sess = None

        if self.network_parameters.framework == Frameworks.TensorFlow:
            general_network = GeneralTensorFlowNetwork
        elif self.network_parameters.framework == Frameworks.Neon:
            general_network = GeneralNeonNetwork
        else:
            raise Exception("{} Framework is not supported"
                            .format(Frameworks().to_string(self.network_parameters.framework)))
        # print(worker_device)
        with tf.variable_scope("{}/{}".format(self.ap.full_name_id, name)):

            # Global network - the main network shared between threads
            self.global_network = None
            if self.has_global:
                # we assign the parameters of this network on the parameters server
                with tf.device(replicated_device):
                    self.global_network = general_network(agent_parameters=agent_parameters,
                                                          name='{}/global'.format(name),
                                                          global_network=None,
                                                          network_is_local=False,
                                                          spaces=spaces,
                                                          network_is_trainable=True)

            # Online network - local copy of the main network used for playing
            self.online_network = None
            with tf.device(worker_device):
                self.online_network = general_network(agent_parameters=agent_parameters,
                                                      name='{}/online'.format(name),
                                                      global_network=self.global_network,
                                                      network_is_local=True,
                                                      spaces=spaces,
                                                      network_is_trainable=True)

            # Target network - a local, slow updating network used for stabilizing the learning
            self.target_network = None
            if self.has_target:
                with tf.device(worker_device):
                    self.target_network = general_network(agent_parameters=agent_parameters,
                                                          name='{}/target'.format(name),
                                                          global_network=self.global_network,
                                                          network_is_local=True,
                                                          spaces=spaces,
                                                          network_is_trainable=False)

            if not isinstance(self.ap.task_parameters, DistributedTaskParameters) and \
                            self.network_parameters.framework == Frameworks.TensorFlow:
                self.model_saver = tf.train.Saver(tf.global_variables())

    def sync(self):
        """
        Initializes the weights of the networks to match each other
        :return:
        """
        self.update_online_network()
        self.update_target_network()

    def update_target_network(self, rate=1.0):
        """
        Copy weights: online network >>> target network
        :param rate: the rate of copying the weights - 1 for copying exactly
        """
        if self.target_network:
            self.target_network.set_weights(self.online_network.get_weights(), rate)

    def update_online_network(self, rate=1.0):
        """
        Copy weights: global network >>> online network
        :param rate: the rate of copying the weights - 1 for copying exactly
        """
        if self.global_network:
            self.online_network.set_weights(self.global_network.get_weights(), rate)

    def apply_gradients_to_global_network(self):
        """
        Apply gradients from the online network on the global network
        :return:
        """
        self.global_network.apply_gradients(self.online_network.accumulated_gradients)

    def apply_gradients_to_online_network(self):
        """
        Apply gradients from the online network on itself
        :return:
        """
        self.online_network.apply_gradients(self.online_network.accumulated_gradients)

    def train_and_sync_networks(self, inputs, targets, additional_fetches=[], importance_weights=None):
        """
        A generic training function that enables multi-threading training using a global network if necessary.
        :param inputs: The inputs for the network.
        :param targets: The targets corresponding to the given inputs
        :param additional_fetches: Any additional tensor the user wants to fetch
        :param importance_weights: A coefficient for each sample in the batch, which will be used to rescale the loss
                                   error of this sample. If it is not given, the samples losses won't be scaled
        :return: The loss of the training iteration
        """
        result = self.online_network.accumulate_gradients(inputs, targets, additional_fetches=additional_fetches,
                                                          importance_weights=importance_weights)
        self.apply_gradients_and_sync_networks()
        return result

    def apply_gradients_and_sync_networks(self):
        """
        Applies the gradients accumulated in the online network to the global network or to itself and syncs the
        networks if necessary
        """
        if self.global_network:
            self.apply_gradients_to_global_network()
            self.online_network.reset_accumulated_gradients()
            self.update_online_network()
        else:
            self.online_network.apply_and_reset_gradients(self.online_network.accumulated_gradients)

    def get_local_variables(self):
        """
        Get all the variables that are local to the thread
        :return: a list of all the variables that are local to the thread
        """
        local_variables = [v for v in tf.local_variables() if self.online_network.name in v.name]
        if self.has_target:
            local_variables += [v for v in tf.local_variables() if self.target_network.name in v.name]
        return local_variables

    def get_global_variables(self):
        """
        Get all the variables that are shared between threads
        :return: a list of all the variables that are shared between threads
        """
        global_variables = [v for v in tf.global_variables() if self.global_network.name in v.name]
        return global_variables

    def set_session(self, sess):
        self.sess = sess
        self.online_network.set_session(sess)
        if self.global_network:
            self.global_network.set_session(sess)
        if self.target_network:
            self.target_network.set_session(sess)

        if self.sess and hasattr(self.ap.task_parameters, 'checkpoint_restore_dir') \
                and self.ap.task_parameters.checkpoint_restore_dir:
            checkpoint = tf.train.latest_checkpoint(self.ap.task_parameters.checkpoint_restore_dir)
            screen.log_title("Loading checkpoint: {}".format(checkpoint))
            self.model_saver.restore(self.sess, checkpoint)
            self.update_target_network()

    def save_model(self, model_id):
        saved_model_path = self.model_saver.save(self.sess,
                                                 os.path.join(self.network_parameters.save_model_dir,
                                                              str(model_id) + '.ckpt'))
        screen.log_dict(
            OrderedDict([
                ("Saving model", saved_model_path),
            ]),
            prefix="Checkpoint"
        )
