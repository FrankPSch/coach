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

from architectures.architecture import Architecture
import tensorflow as tf
from utils import force_list, squeeze_list
from configurations import MiddlewareTypes, AgentParameters
import time
from spaces import SpacesDefinition
from block_factories.block_factory import DistributedTaskParameters
import numpy as np


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        layer_weight_name = '_'.join(var.name.split('/')[-3:])[:-2]

        with tf.name_scope(layer_weight_name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


def local_getter(getter, name, *args, **kwargs):
    """
    This is a wrapper around the tf.get_variable function which puts the variables in the local variables collection
    instead of the global variables collection. The local variables collection will hold variables which are not shared
    between workers. these variables are also assumed to be non-trainable (the optimizer does not apply gradients to
    these variables), but we can calculate the gradients wrt these variables, and we can update their content.
    """
    kwargs['collections'] = [tf.GraphKeys.LOCAL_VARIABLES]
    return getter(name, *args, **kwargs)


class TensorFlowArchitecture(Architecture):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, name: str= "",
                 global_network=None, network_is_local: bool=True, network_is_trainable: bool=False):
        """
        :param agent_parameters: the agent parameters
        :param spaces: the spaces definition of the agent
        :param name: the name of the network
        :param global_network: the global network replica that is shared between all the workers
        :param network_is_local: is the network global (shared between workers) or local (dedicated to the worker)
        :param network_is_trainable: is the network trainable (we can apply gradients on it)
        """
        super().__init__(agent_parameters, spaces, name)
        self.middleware_embedder = None
        self.network_is_local = network_is_local
        if not self.network_parameters.tensorflow_support:
            raise ValueError('TensorFlow is not supported for this agent')
        self.sess = None
        self.inputs = {}
        self.outputs = []
        self.targets = []
        self.importance_weights = []
        self.losses = []
        self.total_loss = None
        self.trainable_weights = []
        self.weights_placeholders = []
        self.curr_rnn_c_in = None
        self.curr_rnn_h_in = None
        self.gradients_wrt_inputs = []
        self.train_writer = None
        self.network_is_trainable = network_is_trainable

        self.optimizer_type = self.network_parameters.optimizer_type
        if self.ap.task_parameters.seed is not None:
            tf.set_random_seed(self.ap.task_parameters.seed)
        with tf.variable_scope("/".join(self.name.split("/")[1:]), initializer=tf.contrib.layers.xavier_initializer(),
                               custom_getter=local_getter if network_is_local and global_network else None):
            self.global_step = tf.train.get_or_create_global_step()

            # build the network
            self.get_model()

            # model weights
            # TODO: why are all the variables going to trainable_variables collection?
            self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.full_name)

            # locks for synchronous training
            # TODO: bring back this if
            # if isinstance(self.ap.task_parameters, DistributedTaskParameters) and not self.network_parameters.async_training \
            #         and not self.network_is_local:
            if not self.network_is_local:
                self.lock_counter = tf.get_variable("lock_counter", [], tf.int32,
                                                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                    trainable=False)
                self.lock = self.lock_counter.assign_add(1, use_locking=True)
                self.lock_init = self.lock_counter.assign(0)

                self.release_counter = tf.get_variable("release_counter", [], tf.int32,
                                                       initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                       trainable=False)
                self.release = self.release_counter.assign_add(1, use_locking=True)
                self.release_init = self.release_counter.assign(0)

            # local network does the optimization so we need to create all the ops we are going to use to optimize
            for idx, var in enumerate(self.weights):
                placeholder = tf.placeholder(tf.float32, shape=var.get_shape(), name=str(idx) + '_holder')
                self.weights_placeholders.append(placeholder)
                if self.ap.visualization.tensorboard:
                    variable_summaries(var)

            self.update_weights_from_list = [weights.assign(holder) for holder, weights in
                                             zip(self.weights_placeholders, self.weights)]

            # gradients ops
            self.tensor_gradients = tf.gradients(self.total_loss, self.weights)
            self.gradients_norm = tf.global_norm(self.tensor_gradients)
            if self.network_parameters.clip_gradients is not None and self.network_parameters.clip_gradients != 0:
                self.clipped_grads, self.grad_norms = tf.clip_by_global_norm(self.tensor_gradients,
                                                                             self.network_parameters.clip_gradients)

            # gradients of the outputs w.r.t. the inputs
            # at the moment, this is only used by ddpg
            self.gradients_wrt_inputs = [{name: tf.gradients(output, input_ph) for name, input_ph in
                                         self.inputs.items()} for output in self.outputs]
            self.gradients_weights_ph = [tf.placeholder('float32', self.outputs[i].shape, 'output_gradient_weights')
                                         for i in range(len(self.outputs))]
            self.weighted_gradients = [tf.gradients(self.outputs[i], self.weights, self.gradients_weights_ph[i])
                                       for i in range(len(self.outputs))]

            # L2 regularization
            if self.network_parameters.l2_regularization != 0:
                self.l2_regularization = [tf.add_n([tf.nn.l2_loss(v) for v in self.weights])
                                          * self.network_parameters.l2_regularization]
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.l2_regularization)

            self.inc_step = self.global_step.assign_add(1)

            # defining the optimization process (for LBFGS we have less control over the optimizer)
            if self.optimizer_type != 'LBFGS' and self.network_is_trainable:
                # no global network, this is a plain simple centralized training
                self.update_weights_from_batch_gradients = self.optimizer.apply_gradients(
                    zip(self.weights_placeholders, self.weights), global_step=self.global_step)

            current_scope_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                        scope=tf.contrib.framework.get_name_scope())
            self.merged = tf.summary.merge(current_scope_summaries)

            # initialize or restore model
            self.init_op = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            )

        self.accumulated_gradients = None

    def set_session(self, sess):
        self.sess = sess

        # initialize the session parameters in single threaded runs. Otherwise, this is done through the
        # MonitoredSession object in the block factory
        if not isinstance(self.ap.task_parameters, DistributedTaskParameters):
            self.sess.run(self.init_op)

            if self.ap.visualization.tensorboard:
                # Write the merged summaries to the current experiment directory
                self.train_writer = tf.summary.FileWriter(self.ap.task_parameters.experiment_path + '/tensorboard',
                                                          self.sess.graph)

        # wait for all the workers to set their session
        if not self.network_is_local:
            self.wait_for_all_workers('lock')  # TODO: currently this won't work properly for >1 agents per task

    def reset_accumulated_gradients(self):
        """
        Reset the gradients accumulation placeholder
        """
        if self.accumulated_gradients is None:
            self.accumulated_gradients = self.sess.run(self.weights)

        for ix, grad in enumerate(self.accumulated_gradients):
            self.accumulated_gradients[ix] = grad * 0

    def accumulate_gradients(self, inputs, targets, additional_fetches=None, importance_weights=None):
        """
        Runs a forward pass & backward pass, clips gradients if needed and accumulates them into the accumulation
        placeholders
        :param additional_fetches: Optional tensors to fetch during gradients calculation
        :param inputs: The input batch for the network
        :param targets: The targets corresponding to the input batch
        :param importance_weights: A coefficient for each sample in the batch, which will be used to rescale the loss
                                   error of this sample. If it is not given, the samples losses won't be scaled
        :return: A list containing the total loss and the individual network heads losses
        """

        if self.accumulated_gradients is None:
            self.reset_accumulated_gradients()

        # feed inputs
        if additional_fetches is None:
            additional_fetches = []
        feed_dict = self._feed_dict(inputs)

        # feed targets
        targets = force_list(targets)
        for placeholder_idx, target in enumerate(targets):
            feed_dict[self.targets[placeholder_idx]] = target

        # feed importance weights
        importance_weights = force_list(importance_weights)
        for placeholder_idx, target_ph in enumerate(targets):
            if len(importance_weights) <= placeholder_idx or importance_weights[placeholder_idx] is None:
                importance_weight = np.ones(targets[placeholder_idx].shape[0])
            else:
                importance_weight = importance_weights[placeholder_idx]
            feed_dict[self.importance_weights[placeholder_idx]] = importance_weight

        if self.optimizer_type != 'LBFGS':
            # set the fetches
            fetches = [self.gradients_norm]
            if self.network_parameters.clip_gradients:
                fetches.append(self.clipped_grads)
            else:
                fetches.append(self.tensor_gradients)
            fetches += [self.total_loss, self.losses]
            if self.network_parameters.middleware_type == MiddlewareTypes.LSTM:
                fetches.append(self.middleware_embedder.state_out)
            additional_fetches_start_idx = len(fetches)
            fetches += additional_fetches

            # feed the lstm state if necessary
            if self.network_parameters.middleware_type == MiddlewareTypes.LSTM:
                # we can't always assume that we are starting from scratch here can we?
                feed_dict[self.middleware_embedder.c_in] = self.middleware_embedder.c_init
                feed_dict[self.middleware_embedder.h_in] = self.middleware_embedder.h_init

            fetches += [self.merged]
            # get grads
            result = self.sess.run(fetches, feed_dict=feed_dict)
            if hasattr(self, 'train_writer') and self.train_writer is not None:
                self.train_writer.add_summary(result[-1], self.ap.current_episode)

            # extract the fetches
            norm_unclipped_grads, grads, total_loss, losses = result[:4]
            if self.network_parameters.middleware_type == MiddlewareTypes.LSTM:
                (self.curr_rnn_c_in, self.curr_rnn_h_in) = result[4]
            fetched_tensors = []
            if len(additional_fetches) > 0:
                fetched_tensors = result[additional_fetches_start_idx:additional_fetches_start_idx +
                                                                      len(additional_fetches)]

            # accumulate the gradients
            for idx, grad in enumerate(grads):
                self.accumulated_gradients[idx] += grad

            return total_loss, losses, norm_unclipped_grads, fetched_tensors

        else:
            self.optimizer.minimize(session=self.sess, feed_dict=feed_dict)

            return [0]

    def _feed_dict(self, inputs):
        feed_dict = {}
        for input_name, input_value in inputs.items():
            if isinstance(input_name, str):
                if input_name not in self.inputs:
                    raise ValueError((
                        'input name {input_name} was provided to create a feed '
                        'dictionary, but there is no placeholder with that name. '
                        'placeholder names available include: {placeholder_names}'
                    ).format(
                        input_name=input_name,
                        placeholder_names=', '.join(self.inputs.keys())
                    ))

                feed_dict[self.inputs[input_name]] = input_value
            elif isinstance(input_name, tf.Tensor) and input_name.op.type == 'Placeholder':
                feed_dict[input_name] = input_value
            else:
                raise ValueError((
                    'input dictionary expects strings or placeholders as keys, '
                    'but found key {key} of type {type}'
                ).format(
                    key=input_name,
                    type=type(input_name),
                ))

        return feed_dict

    def apply_and_reset_gradients(self, gradients, scaler=1.):
        """
        Applies the given gradients to the network weights and resets the accumulation placeholder
        :param gradients: The gradients to use for the update
        :param scaler: A scaling factor that allows rescaling the gradients before applying them
        """
        self.apply_gradients(gradients, scaler)
        self.reset_accumulated_gradients()

    def wait_for_all_workers(self, lock: str):
        """
        A barrier that allows waiting for all the workers to finish a certain block of commands
        :param lock: the name of the lock to use
        :return: None
        """
        # TODO: try to move this function up in the hierarchy
        # lock
        if hasattr(self, '{}_counter'.format(lock)):
            self.sess.run(getattr(self, lock))
            while self.sess.run(getattr(self, '{}_counter'.format(lock))) % self.ap.task_parameters.num_tasks != 0:
                time.sleep(0.00001)
        else:
            raise ValueError("no counter was defined for the lock {}".format(lock))

    def apply_gradients(self, gradients, scaler=1.):
        """
        Applies the given gradients to the network weights
        :param gradients: The gradients to use for the update
        :param scaler: A scaling factor that allows rescaling the gradients before applying them
        """
        if self.network_parameters.async_training or not isinstance(self.ap.task_parameters, DistributedTaskParameters):
            if hasattr(self, 'global_step') and not self.network_is_local:
                self.sess.run(self.inc_step)

        if self.optimizer_type != 'LBFGS':

            # lock barrier
            # TODO: use wait_for_all_workers
            if hasattr(self, 'lock_counter'):
                self.sess.run(self.lock)
                while self.sess.run(self.lock_counter) % self.ap.task_parameters.num_training_tasks != 0:
                    time.sleep(0.00001)
                # rescale the gradients so that they average out with the gradients from the other workers
                scaler /= float(self.ap.task_parameters.num_training_tasks)

            # apply gradients
            if scaler != 1.:
                for gradient in gradients:
                    gradient /= scaler
            feed_dict = dict(zip(self.weights_placeholders, gradients))

            _ = self.sess.run(self.update_weights_from_batch_gradients, feed_dict=feed_dict)

            # release barrier
            if hasattr(self, 'release_counter'):
                self.sess.run(self.release)
                while self.sess.run(self.release_counter) % self.ap.task_parameters.num_training_tasks != 0:
                    time.sleep(0.00001)

    def predict(self, inputs, outputs=None, squeeze_output=True):
        """
        Run a forward pass of the network using the given input
        :param inputs: The input for the network
        :param outputs: The output for the network, defaults to self.outputs
        :param squeeze_output: call squeeze_list on output
        :return: The network output

        WARNING: must only call once per state since each call is assumed by LSTM to be a new time step.
        """

        feed_dict = self._feed_dict(inputs)
        if outputs is None:
            outputs = self.outputs

        if self.network_parameters.middleware_type == MiddlewareTypes.LSTM:
            feed_dict[self.middleware_embedder.c_in] = self.curr_rnn_c_in
            feed_dict[self.middleware_embedder.h_in] = self.curr_rnn_h_in

            output, (self.curr_rnn_c_in, self.curr_rnn_h_in) = self.sess.run([outputs, self.middleware_embedder.state_out], feed_dict=feed_dict)
        else:
            output = self.sess.run(outputs, feed_dict)

        if squeeze_output:
            output = squeeze_list(output)
        return output

    def train_on_batch(self, inputs, targets, scaler=1., additional_fetches=None, importance_weights=None):
        """
        Given a batch of examples and targets, runs a forward pass & backward pass and then applies the gradients
        :param additional_fetches: Optional tensors to fetch during the training process
        :param inputs: The input for the network
        :param targets: The targets corresponding to the input batch
        :param scaler: A scaling factor that allows rescaling the gradients before applying them
        :param importance_weights: A coefficient for each sample in the batch, which will be used to rescale the loss
                                   error of this sample. If it is not given, the samples losses won't be scaled
        :return: The loss of the network
        """
        if additional_fetches is None:
            additional_fetches = []
        force_list(additional_fetches)
        loss = self.accumulate_gradients(inputs, targets, additional_fetches=additional_fetches,
                                         importance_weights=importance_weights)
        self.apply_and_reset_gradients(self.accumulated_gradients, scaler)
        return loss

    def get_weights(self):
        """
        :return: a list of tensors containing the network weights for each layer
        """
        return self.weights

    def set_weights(self, weights, new_rate=1.0):
        """
        Sets the network weights from the given list of weights tensors
        """
        feed_dict = {}
        old_weights, new_weights = self.sess.run([self.get_weights(), weights])
        for placeholder_idx, new_weight in enumerate(new_weights):
            feed_dict[self.weights_placeholders[placeholder_idx]]\
                = new_rate * new_weight + (1 - new_rate) * old_weights[placeholder_idx]
        self.sess.run(self.update_weights_from_list, feed_dict)

    def get_variable_value(self, variable):
        """
        Get the value of a variable from the graph
        :param variable: the variable
        :return: the value of the variable
        """
        return self.sess.run(variable)

    def set_variable_value(self, assign_op, value, placeholder=None):
        """
        Updates the value of a variable.
        This requires having an assign operation for the variable, and a placeholder which will provide the value
        :param assign_op: an assign operation for the variable
        :param value: a value to set the variable to
        :param placeholder: a placeholder to hold the given value for injecting it into the variable
        """
        self.sess.run(assign_op, feed_dict={placeholder: value})
