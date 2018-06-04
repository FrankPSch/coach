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

try:
    import matplotlib.pyplot as plt
except:
    from logger import failed_imports
    failed_imports.append("matplotlib")

from utils import Signal, RunningStat, squeeze_list, force_list
from core_types import RunPhase, PredictionType, Episodes
from configurations import MiddlewareTypes

import copy
import numpy as np
from spaces import SpacesDefinition
from logger import screen, Logger, EpisodeLogger
import random
from six.moves import range
from agents.agent_interface import AgentInterface
from core_types import Transition, ActionInfo, TrainingSteps, EnvironmentSteps, EnvResponse
from utils import dynamic_import_and_instantiate_module_from_params, call_method_for_all
from collections import OrderedDict
from pandas import read_pickle
from configurations import AgentParameters
from typing import Dict, List, Union, Tuple
from architectures.network_wrapper import NetworkWrapper
from block_factories.block_factory import DistributedTaskParameters
from memories.episodic_experience_replay import EpisodicExperienceReplay


class Agent(AgentInterface):
    def __init__(self, agent_parameters: AgentParameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        """
        :param agent_parameters: A Preset class instance with all the running paramaters
        """
        super().__init__()
        self.ap = agent_parameters
        self.task_id = self.ap.task_parameters.task_index
        self.is_chief = self.task_id == 0
        if self.ap.memory.distributed_memory:
            self.shared_memory_scratchpad = self.ap.task_parameters.shared_memory_scratchpad
        self.name = agent_parameters.name
        self.parent = parent
        self.parent_level_manager = None
        # self.full_name_id = agent_parameters.full_name_id = '/'.join([#self.parent.parent_level_manager.name,
        #                                                               #self.parent.name,
        #                                                               self.name])
        self.full_name_id = agent_parameters.full_name_id = self.name

        if type(agent_parameters.task_parameters) == DistributedTaskParameters:
            screen.log_title("Creating agent - name: {} task id: {} (may take up to 30 seconds due to "
                             "tensorflow wake up time)".format(self.full_name_id, self.task_id))
        else:
            screen.log_title("Creating agent - name: {}".format(self.full_name_id))
        self.imitation = False
        self.agent_logger = Logger()
        self.agent_episode_logger = EpisodeLogger()

        # i/o dimensions
        # TODO: update everyone that uses  desired_observation_width, action_space_size and measurements_size
        # + update the measurements_size if use_accumulated_reward_as_measurement is used

        memory_name = self.ap.memory.path.split(':')[1]
        lookup_name = self.full_name_id + '.' + memory_name
        if self.ap.memory.distributed_memory is True and not self.is_chief:
            self.memory = self.shared_memory_scratchpad.get(lookup_name)
            # print("agent {} just got a {} from the memory scratchpad".format(self.task_id, memory_name))
        else:
            # modules
            if agent_parameters.memory.load_memory_from_file_path:
                screen.log_title("Loading replay buffer from pickle. Pickle path: {}"
                                 .format(agent_parameters.memory.load_memory_from_file_path))
                self.memory = read_pickle(agent_parameters.memory.load_memory_from_file_path)
            else:
                self.memory = dynamic_import_and_instantiate_module_from_params(self.ap.memory)

            if self.ap.memory.distributed_memory is True and self.is_chief:
                self.shared_memory_scratchpad.add(lookup_name, self.memory)
                # print("agent {} just pushed a {} to the memory scratchpad".format(self.task_id, memory_name))

        if type(agent_parameters.task_parameters) == DistributedTaskParameters:
            self.has_global = True
            self.replicated_device = agent_parameters.task_parameters.device
            self.worker_device = "/job:worker/task:{}/cpu:0".format(self.task_id)
        else:
            self.has_global = False
            self.replicated_device = None
            self.worker_device = "/gpu:0"

        # filters
        self.input_filter = self.ap.input_filter
        self.output_filter = self.ap.output_filter
        device = self.replicated_device if self.replicated_device else self.worker_device
        self.input_filter.set_device(device)
        self.output_filter.set_device(device)

        # initialize all internal variables
        self._phase = RunPhase.HEATUP
        self.total_shaped_reward_in_current_episode = 0
        self.total_reward_in_current_episode = 0
        self.total_steps_counter = 0
        self.running_reward = None
        self.training_iteration = 0
        self.last_target_network_update_step = 0
        self.last_training_phase_step = 0
        self.current_episode = self.ap.current_episode = 0
        self.curr_state = {}
        self.current_episode_steps_counter = 0
        self.episode_running_info = {}
        self.last_episode_evaluation_ran = 0
        self.running_observations = []
        self.agent_logger.set_current_time(self.current_episode)
        self.exploration_policy = None
        self.networks = {}
        self.last_action_info = None
        self.running_observation_stats = None
        self.running_reward_stats = None
        # TODO: add observations rendering from master

        # environment parameters
        self.spaces = None

        # signals
        self.episode_signals = []
        self.step_signals = []
        self.loss = self.register_signal('Loss')
        self.curr_learning_rate = self.register_signal('Learning Rate')
        self.unclipped_grads = self.register_signal('Grads (unclipped)')
        self.reward = self.register_signal('Reward', dump_one_value_per_episode=False, dump_one_value_per_step=True)
        self.shaped_reward = self.register_signal('Shaped Reward', dump_one_value_per_episode=False, dump_one_value_per_step=True)

        # TODO - reenable this
        # if self.ap.env.check_successes:
        #     self.success_ratio = self.register_signal('Success Ratio')
        #     self.mean_reward = self.register_signal('Mean Reward')

        # use seed
        if self.ap.task_parameters.seed is not None:
            random.seed(self.ap.task_parameters.seed)
            np.random.seed(self.ap.task_parameters.seed)

    def setup_logger(self):
        # TODO: this is ugly. we should do it nicer.
        # dump documentation
        logger_prefix = "{block_name}.{level_name}.{agent_full_id}".\
            format(block_name=self.parent_level_manager.parent_block_scheduler.name,
                   level_name=self.parent_level_manager.name,
                   agent_full_id='.'.join(self.full_name_id.split('/')))
        self.agent_logger.set_logger_filenames(self.ap.task_parameters.experiment_path, logger_prefix=logger_prefix,
                                               add_timestamp=True, task_id=self.task_id)
        if self.ap.visualization.dump_in_episode_signals:
            self.agent_episode_logger.set_logger_filenames(self.ap.task_parameters.experiment_path,
                                                           logger_prefix=logger_prefix,
                                                           add_timestamp=True, task_id=self.task_id)

    def set_session(self, sess) -> None:
        """
        Set the deep learning framework session for all the agents in the composite agent
        :return: None
        """
        self.input_filter.set_session(sess)
        self.output_filter.set_session(sess)
        [network.set_session(sess) for network in self.networks.values()]

    def register_signal(self, signal_name: str, dump_one_value_per_episode: bool=True,
                        dump_one_value_per_step: bool=False) -> Signal:
        signal = Signal(signal_name)
        if dump_one_value_per_episode:
            self.episode_signals.append(signal)
        if dump_one_value_per_step:
            self.step_signals.append(signal)
        return signal

    def set_environment_parameters(self, spaces: SpacesDefinition):
        """
        Sets the parameters that are environment dependent. As a side effect, initializes all the components that are
        dependent on those values, by calling init_environment_dependent_modules
        :param spaces: the environment spaces definition
        :return: None
        """
        self.spaces = copy.deepcopy(spaces)
        for observation_name in self.spaces.state.sub_spaces.keys():
            self.spaces.state[observation_name] = \
                self.input_filter.get_filtered_observation_space(observation_name, self.spaces.state[observation_name])
        self.spaces.reward = self.input_filter.get_filtered_reward_space(self.spaces.reward)
        self.spaces.action = self.output_filter.get_unfiltered_action_space(self.spaces.action)
        self.init_environment_dependent_modules()

    def create_networks(self) -> Dict[str, NetworkWrapper]:
        """
        Create all the networks of the agent.
        The network creation will be done after setting the environment parameters for the agent, since they are needed
        for creating the network.
        :return: A list containing all the networks
        """
        networks = {}
        for network_name in self.ap.network_wrappers.keys():
            networks[network_name] = NetworkWrapper(name=network_name,
                                                    agent_parameters=self.ap,
                                                    has_target=self.ap.network_wrappers[network_name].create_target_network,
                                                    has_global=self.has_global,
                                                    spaces=self.spaces,
                                                    replicated_device=self.replicated_device,
                                                    worker_device=self.worker_device)
        return networks

    def init_environment_dependent_modules(self) -> None:
        """
        Initialize any modules that depend on knowing information about the environment such as the action space or
        the observation space
        :return: None
        """
        # initialize exploration policy
        self.ap.exploration.action_space = self.spaces.action
        self.exploration_policy = dynamic_import_and_instantiate_module_from_params(self.ap.exploration)

        # create all the networks of the agent
        self.networks = self.create_networks()

    @property
    def phase(self) -> RunPhase:
        return self._phase

    @phase.setter
    def phase(self, val: RunPhase) -> None:
        """
        Change the phase of the run for the agent and all the sub components
        :param phase: the new run phase (TRAIN, TEST, etc.)
        :return: None
        """
        self._phase = val
        self.exploration_policy.change_phase(val)

    def log_to_screen(self):
        # log to screen
        log = OrderedDict()
        if self.task_id is not None:
            log["Worker"] = self.task_id
        log["Episode"] = self.current_episode
        log["Total reward"] = np.round(self.total_reward_in_current_episode, 2)
        log["Exploration"] = np.round(self.exploration_policy.get_control_param(), 2)
        log["Steps"] = self.total_steps_counter
        log["Training iteration"] = self.training_iteration
        screen.log_dict(log, prefix=self.phase.value)

    def update_step_in_episode_log(self):
        """
        Writes logging messages to screen and updates the log file with all the signal values.
        :return: None
        """
        # log all the signals to file
        self.agent_episode_logger.set_current_time(self.current_episode_steps_counter)
        self.agent_episode_logger.create_signal_value('Training Iter', self.training_iteration)
        self.agent_episode_logger.create_signal_value('In Heatup', int(self._phase == RunPhase.HEATUP))
        self.agent_episode_logger.create_signal_value('ER #Transitions', self.memory.num_transitions())
        self.agent_episode_logger.create_signal_value('ER #Episodes', self.memory.length())
        self.agent_episode_logger.create_signal_value('Total steps', self.total_steps_counter)
        self.agent_episode_logger.create_signal_value("Epsilon", self.exploration_policy.get_control_param())
        self.agent_episode_logger.create_signal_value("Shaped Accumulated Reward", self.total_shaped_reward_in_current_episode)
        self.agent_episode_logger.create_signal_value('Update Target Network', 0, overwrite=False)
        self.agent_episode_logger.update_wall_clock_time(self.current_episode_steps_counter)

        for signal in self.step_signals:
            self.agent_episode_logger.create_signal_value(signal.name, signal.get_last_value())

        # dump
        self.agent_episode_logger.dump_output_csv()

    def update_log(self):
        """
        Writes logging messages to screen and updates the log file with all the signal values.
        :return: None
        """
        # log all the signals to file
        self.agent_logger.set_current_time(self.current_episode)
        self.agent_logger.create_signal_value('Training Iter', self.training_iteration)
        self.agent_logger.create_signal_value('In Heatup', int(self._phase == RunPhase.HEATUP))
        self.agent_logger.create_signal_value('ER #Transitions', self.memory.num_transitions())
        self.agent_logger.create_signal_value('ER #Episodes', self.memory.length())
        self.agent_logger.create_signal_value('Episode Length', self.current_episode_steps_counter)
        self.agent_logger.create_signal_value('Total steps', self.total_steps_counter)
        self.agent_logger.create_signal_value("Epsilon", np.mean(self.exploration_policy.get_control_param()))
        self.agent_logger.create_signal_value("Shaped Training Reward", self.total_shaped_reward_in_current_episode
                                   if self._phase == RunPhase.TRAIN else np.nan)
        self.agent_logger.create_signal_value("Training Reward", self.total_reward_in_current_episode
                                   if self._phase == RunPhase.TRAIN else np.nan)
        self.agent_logger.create_signal_value('Shaped Evaluation Reward', self.total_shaped_reward_in_current_episode
                                   if self._phase == RunPhase.TEST else np.nan)
        self.agent_logger.create_signal_value('Evaluation Reward', self.total_reward_in_current_episode
                                   if self._phase == RunPhase.TEST else np.nan)
        self.agent_logger.create_signal_value('Update Target Network', 0, overwrite=False)
        self.agent_logger.update_wall_clock_time(self.current_episode)

        for signal in self.episode_signals:
            self.agent_logger.create_signal_value("{}/Mean".format(signal.name), signal.get_mean())
            self.agent_logger.create_signal_value("{}/Stdev".format(signal.name), signal.get_stdev())
            self.agent_logger.create_signal_value("{}/Max".format(signal.name), signal.get_max())
            self.agent_logger.create_signal_value("{}/Min".format(signal.name), signal.get_min())

        # dump
        if self.current_episode % self.ap.visualization.dump_signals_to_csv_every_x_episodes == 0 \
                and self.current_episode > 0:
            self.agent_logger.dump_output_csv()

    def end_episode(self) -> None:
        """
        End an episode
        :return: None
        """
        self.current_episode += 1
        if self.ap.visualization.dump_csv:
            self.update_log()

        if self.ap.visualization.print_summary:
            self.log_to_screen()

    def reset(self):
        """
        Reset all the episodic parameters
        :return: None
        """
        for signal in self.episode_signals:
            signal.reset()
        for signal in self.step_signals:
            signal.reset()
        self.agent_episode_logger.set_episode_idx(self.current_episode)
        self.total_shaped_reward_in_current_episode = 0
        self.total_reward_in_current_episode = 0
        self.curr_state = {}
        self.current_episode_steps_counter = 0
        self.episode_running_info = {}
        if self.exploration_policy:
            self.exploration_policy.reset()
        self.input_filter.reset()
        self.output_filter.reset()
        if isinstance(self.memory, EpisodicExperienceReplay):
            self.memory.verify_last_episode_is_closed()

        #TODO - this is broken, need to match network name in ap to the agent's networks (i.e. have dict instead of
        #  list)
        for network_wrapper in self.ap.network_wrappers.keys():
            if self.ap.network_wrappers[network_wrapper].middleware_type == MiddlewareTypes.LSTM:
                network = self.networks[network_wrapper]
                network.online_network.curr_rnn_c_in = network.online_network.middleware_embedder.c_init
                network.online_network.curr_rnn_h_in = network.online_network.middleware_embedder.h_init

    def learn_from_batch(self, batch) -> Tuple[float, List, List]:
        """
        Given a batch of transitions, calculates their target values and updates the network.
        :param batch: A list of transitions
        :return: The total loss of the training, the loss per head and the unclipped gradients
        """
        return 0, [], []

    def _should_update_online_weights_to_target(self):
        """
        Determine if online weights should be copied to the target.
        :return: boolean: True if the online weights should be copied to the target.
        """
        # TODO: modify all the presets such that this parameter will be defined in this manner
        # TODO: this shouldn't be called if there is no target network
        # TODO: allow specifying in frames
        # update the target network of every network that has a target network
        step_method = self.ap.algorithm.num_steps_between_copying_online_weights_to_target
        if step_method.__class__ == TrainingSteps:
            should_update = (self.training_iteration - self.last_target_network_update_step) >= step_method.num_steps
            if should_update:
                self.last_target_network_update_step = self.training_iteration
        elif step_method.__class__ == EnvironmentSteps:
            should_update = (self.total_steps_counter - self.last_target_network_update_step) >= step_method.num_steps
            if should_update:
                self.last_target_network_update_step = self.total_steps_counter
        else:
            raise ValueError("The num_steps_between_copying_online_weights_to_target parameter should be either "
                             "EnvironmentSteps or TrainingSteps. Instead it is {}".format(step_method.__class__))
        return should_update

    def _should_train(self, wait_for_full_episode=False):
        """
        Determine if we should start a training phase according to the number of steps passed since the last training
        :return:  boolean: True if we should start a training phase
        """
        step_method = self.ap.algorithm.num_consecutive_playing_steps
        if step_method.__class__ == Episodes:
            should_update = (self.current_episode - self.last_training_phase_step) >= step_method.num_steps
            if should_update:
                self.last_training_phase_step = self.current_episode
        elif step_method.__class__ == EnvironmentSteps:
            should_update = (self.total_steps_counter - self.last_training_phase_step) >= step_method.num_steps
            if wait_for_full_episode:
                should_update = should_update and self.current_episode_steps_counter == 0
            if should_update:
                self.last_training_phase_step = self.total_steps_counter
        else:
            raise ValueError("The num_consecutive_playing_steps parameter should be either "
                             "EnvironmentSteps or Episodes. Instead it is {}".format(step_method.__class__))
        return should_update

    def train(self):
        """
        Check if a training phase should be done as configured by num_consecutive_playing_steps.
        If it should, then do several training steps as configured by num_consecutive_training_steps.
        A single training iteration: Sample a batch, train on it and update target networks.
        :return: The total training loss during the training iterations.
        """
        loss = 0
        if self._should_train():
            for training_step in range(self.ap.algorithm.num_consecutive_training_steps):
                # TODO: this should be network dependent!
                network_parameters = list(self.ap.network_wrappers.values())[0]

                # update counters
                self.training_iteration += 1

                # sample a batch and train on it
                batch = self.memory.sample(network_parameters.batch_size)

                # if the batch returned empty then there are not enough samples in the replay buffer -> skip
                # training step
                if len(batch) > 0:
                    # train
                    total_loss, losses, unclipped_grads = self.learn_from_batch(batch)
                    loss += total_loss
                    self.unclipped_grads.add_sample(unclipped_grads)

                    # TODO: why is this done here? it also uses tensorflow directly which is problematic
                    # decay learning rate
                    if network_parameters.learning_rate_decay_rate != 0:
                        self.curr_learning_rate.add_sample(self.networks['main'].sess.run(
                            self.networks['main'].online_network.current_learning_rate))
                    else:
                        self.curr_learning_rate.add_sample(network_parameters.learning_rate)

                    if self._should_update_online_weights_to_target():
                        for network in self.networks.values():
                            network.update_target_network(self.ap.algorithm.rate_for_copying_weights_to_target)

                        self.agent_logger.create_signal_value('Update Target Network', 1)
                    else:
                        self.agent_logger.create_signal_value('Update Target Network', 0, overwrite=False)

                    self.loss.add_sample(loss)

                    if self.imitation:
                        self.log_to_screen()

            # run additional commands after the training is done
            self.post_training_commands()

        return loss

    def extract_batch(self, batch, network_name):
        """
        Extracts a single numpy array for each object in a batch of transitions (state, action, etc.)
        :param batch: An array of transitions
        :return: For each transition element, returns a numpy array of all the transitions in the batch
        """

        # extract current and next states
        current_states = {}
        next_states = {}
        for key in self.ap.network_wrappers[network_name].input_types.keys():
            current_states[key] = np.array([np.array(transition.state[key]) for transition in batch])
            next_states[key] = np.array([np.array(transition.next_state[key]) for transition in batch])

        # extract the rest of the data into batch structures
        actions = np.array([transition.action for transition in batch])
        rewards = np.array([transition.reward for transition in batch])
        game_overs = np.array([transition.game_over for transition in batch])

        # check if the total return is available
        total_return = None
        if batch[0]._total_return:
            total_return = np.array([transition.total_return for transition in batch])

        return current_states, next_states, actions, rewards, game_overs, total_return

    def choose_action(self, curr_state):
        """
        choose an action to act with in the current episode being played. Different behavior might be exhibited when training
         or testing.

        :param curr_state: the current state to act upon.
        :return: chosen action, some action value describing the action (q-value, probability, etc)
        """
        pass

    def dict_state_to_batches_dict(self, curr_state: Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]],
                                   network_name: str):
        """
        convert curr_state into input tensors tensorflow is expecting.
        """
        # convert to batch so we can run it through the network
        curr_states = force_list(curr_state)
        batches_dict = {}
        for key in self.ap.network_wrappers[network_name].input_types.keys():
            if key in curr_states[0].keys():
                batches_dict[key] = np.array([np.array(curr_state[key]) for curr_state in curr_states])

        return batches_dict

    def act(self) -> ActionInfo:
        """
        Given the agents current knowledge, decide on the next action to apply to the environment
        :return: an action and a dictionary containing any additional info from the action decision process
        """
        # assert type(self.ap.algorithm.num_consecutive_playing_steps) == EnvironmentSteps
        if self.phase == RunPhase.TRAIN and self.ap.algorithm.num_consecutive_playing_steps.num_steps == 0:
            # This agent never plays  while training (e.g. behavioral cloning)
            return None

        # count steps (only when training or if we are in the evaluation worker)
        if self.phase != RunPhase.TEST or self.ap.task_parameters.evaluate_only:
            self.total_steps_counter += 1
        self.current_episode_steps_counter += 1

        # decide on the action
        if self.phase == RunPhase.HEATUP and not self.ap.algorithm.heatup_using_network_decisions:
            # random action
            self.last_action_info = self.spaces.action.sample_with_info()
        else:
            # informed action
            self.last_action_info = self.choose_action(self.curr_state)

        filtered_action_info = self.output_filter.filter(self.last_action_info)

        return filtered_action_info

    def get_state_embedding(self, state: dict) -> np.ndarray:
        """
        Given a state, get the corresponding state embedding  from the main network
        :param state: a state dict
        :return: a numpy embedding vector
        """
        # TODO: this won't work anymore
        # TODO: instead of the state embedding (which contains the goal) we should use the observation embedding
        embedding = self.networks['main'].online_network.predict(
            self.dict_state_to_batches_dict(state, "main"),
            outputs=self.networks['main'].online_network.state_embedding)
        return embedding

    def observe(self, env_response: EnvResponse) -> bool:
        """
        Given a response from the environment, distill the observation from it and store it for later use.
        The response should be a dictionary containing the performed action, the new observation and measurements,
        the reward, a game over flag and any additional information necessary.
        :param env_response: result of call from environment.step(action)
        :return:
        """

        # filter the env_response
        filtered_env_response = self.input_filter.filter(env_response)

        if self.current_episode_steps_counter > 0:
            transition = Transition(state=copy.copy(self.curr_state), action=self.last_action_info.action,
                                    reward=filtered_env_response.reward, next_state=filtered_env_response.new_state,
                                    game_over=filtered_env_response.game_over, info=filtered_env_response.info,
                                    goal=filtered_env_response.goal)
        else:
            transition = Transition(next_state=filtered_env_response.new_state,
                                    game_over=filtered_env_response.game_over,
                                    info=filtered_env_response.info,
                                    goal=filtered_env_response.goal)

        self.curr_state = transition.next_state

        # if we are in the first step in the episode, then we don't have a transition yet, and therefore we don't need
        # to store anything in the memory, and we have no reward
        if self.current_episode_steps_counter > 0:
            # merge the intrinsic reward in
            if self.ap.algorithm.scale_external_reward_by_intrinsic_reward_value:
                transition.reward = transition.reward * (1 + self.last_action_info.action_intrinsic_reward)
            else:
                transition.reward = transition.reward + self.last_action_info.action_intrinsic_reward

            # TODO: we want to also have the raw reward for printing
            # sum up the total shaped reward
            self.total_shaped_reward_in_current_episode += transition.reward
            self.total_reward_in_current_episode += env_response.reward
            self.shaped_reward.add_sample(transition.reward)
            self.reward.add_sample(env_response.reward)

            # add action info to transition
            if type(self.parent).__name__ == 'CompositeAgent':
                transition.add_info(self.parent.last_action_info.__dict__)
            else:
                transition.add_info(self.last_action_info.__dict__)

            # TODO: implement as a filter
            if self.ap.algorithm.use_accumulated_reward_as_measurement:
                transition.next_state['measurements'] = np.append(transition.next_state['measurements'],
                                                                  self.total_shaped_reward_in_current_episode)
            # TODO: move to a filter
            if self.ap.algorithm.add_a_normalized_timestep_to_the_observation:
                transition.info['timestep'] = float(self.current_episode_steps_counter) / self.env.timestep_limit

            # create and store the transition
            if self.phase in [RunPhase.TRAIN, RunPhase.HEATUP]:
                self.memory.store(transition)

            if self.ap.visualization.dump_in_episode_signals:
                self.update_step_in_episode_log()

        return transition.game_over

    # def evaluate(self, num_episodes, keep_networks_synced=False):
    #     """
    #     Run in an evaluation mode for several episodes. Actions will be chosen greedily.
    #     :param keep_networks_synced: keep the online network in sync with the global network after every episode
    #     :param num_episodes: The number of episodes to evaluate on
    #     :return: None
    #     """
    #
    #     max_reward_achieved = -float('inf')
    #     average_evaluation_reward = 0
    #
    #     if self.ap.env.check_successes:
    #         number_successes = 0
    #
    #     screen.log_title("Running evaluation")
    #     self.phase = RunPhase.TEST
    #     for i in range(num_episodes):
    #         # keep the online network in sync with the global network
    #         if keep_networks_synced:
    #             for network in self.networks.values():
    #                 network.sync()
    #
    #         # act for one episode
    #         episode_ended = False
    #         while not episode_ended:
    #             episode_ended = self.act_and_observe()
    #
    #             if keep_networks_synced \
    #                and self.total_steps_counter % self.ap.algorithm.update_evaluation_agent_network_after_every_num_steps:
    #                 for network in self.networks.values():
    #                     network.sync()
    #
    #         average_evaluation_reward += self.total_shaped_reward_in_current_episode
    #
    #         # check if the reward passed the reward threshold
    #         if self.ap.env.check_successes:
    #             if self.ap.env.custom_reward_threshold is not None:
    #                 reward_threshold = self.ap.env.custom_reward_threshold
    #             elif self.env.reward_success_threshold is not None:
    #                 reward_threshold = self.env.reward_success_threshold
    #             else:
    #                 raise ValueError("There is no reward threshold defined for the environment")
    #             if self.total_shaped_reward_in_current_episode >= reward_threshold:
    #                 number_successes += 1
    #
    #         self.reset()
    #
    #     # summarize the evaluation phase
    #     average_evaluation_reward /= float(num_episodes)
    #     screen.log_title("Evaluation done. Average reward = {}.".format(average_evaluation_reward))
    #
    #     # TODO: determine general method for allowing users to specify custom
    #     # metrics/callbacks
    #     if self.ap.env.check_successes:
    #         percent_success = number_successes / float(num_episodes)
    #         screen.log_title("Percent success = {}.".format(percent_success))
    #         self.success_ratio.add_sample(percent_success)
    #
    #         if percent_success >= 1.0:
    #             # end training
    #             self.ap.num_training_iterations = self.training_iteration
    #
    #         # TODO: add a flag for this
    #         mean_reward = self.memory.mean_reward()
    #         screen.log_title("mean_reward = {}.".format(mean_reward))
    #         self.mean_reward.add_sample(mean_reward)
    #
    #     self.phase = RunPhase.TRAIN

    def post_training_commands(self):
        pass

    def save_checkpoint(self, model_id: str="checkpoint"):
        # TODO: define this as a global saving function and not per network saving function
        if self.ap.task_parameters.save_model_dir:
            list(self.networks.values())[0].network_parameters.save_model_dir = self.ap.task_parameters.save_model_dir
            list(self.networks.values())[0].save_model(model_id)

    def get_predictions(self, states: List[Dict[str, np.ndarray]], prediction_type: PredictionType):
        """
        Get a prediction from the agent with regard to the requested prediction_type.
        If the agent cannot predict this type of prediction_type, or if there is more than possible way to do so,
        raise a ValueException.
        :param states:
        :param prediction_type:
        :return:
        """

        predictions = self.networks['main'].online_network.predict_with_prediction_type(
            # states=self.dict_state_to_batches_dict(states, 'main'), prediction_type=prediction_type)
            states=states, prediction_type=prediction_type)

        if len(predictions.keys()) != 1:
            raise ValueError("The network has more than one component {} matching the requested prediction_type {}. ".
                             format(list(predictions.keys()), prediction_type))
        return list(predictions.values())[0]

    def sync(self) -> None:
        """
        Sync the global network parameters to local networks
        :return: None
        """
        for network in self.networks.values():
            network.sync()





