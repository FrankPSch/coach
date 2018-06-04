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


import argparse

from level_manager import LevelManager
from typing import List
from core_types import StepMethod, RunPhase, TrainingSteps, PlayingStepsType, Episodes, Frames, EnvironmentSteps
from logger import screen, Logger
import time
from utils import short_dynamic_import, call_method_for_all
from environments.environment import Environment, LevelSelection
from configurations import Parameters
import sys
from core_types import *


class BlockSchedulerParameters(Parameters):
    def __init__(self):
        super().__init__()
        self.heatup_steps = None
        self.evaluation_steps = None
        self.steps_between_evaluation_periods = None
        self.improve_steps = None


class HumanPlayBlockSchedulerParameters(BlockSchedulerParameters):
    def __init__(self):
        super().__init__()
        self.heatup_steps = EnvironmentSteps(0)
        self.evaluation_steps = Episodes(0)
        self.steps_between_evaluation_periods = Episodes(100000000)
        self.improve_steps = TrainingSteps(10000000000)


class BlockScheduler(object):
    """
    A block scheduler is a wrapper around all the hierarchy levels within a single process.
    The block scheduler acts as a scheduler and controls the run in terms of when to:
    - do heatup
    - do training
    - evaluate the model
    - save checkpoints
    For each phase, it defines how many steps to perform, and it loops the different hierarchy levels and
    calls their corresponding methods.
    """
    def __init__(self,
                 name: str,
                 level_managers: List[LevelManager],
                 environments: List[Environment],
                 heatup_steps: PlayingStepsType,
                 evaluation_steps: PlayingStepsType,
                 steps_between_evaluation_periods: StepMethod,
                 improve_steps: StepMethod,
                 task_parameters: 'TaskParameters'):
        super().__init__()
        self.level_managers = level_managers
        self.top_level_manager = level_managers[0]
        self.environments = environments
        self.heatup_steps = heatup_steps
        self.evaluation_steps = evaluation_steps
        self.steps_between_evaluation_periods = steps_between_evaluation_periods
        self.improve_steps = improve_steps
        self.name = name
        self.task_parameters = task_parameters
        self.parent_block_factory = None
        self._phase = self.phase = RunPhase.UNDEFINED

        # timers
        self.block_initialization_time = time.time()
        self.heatup_start_time = None
        self.training_start_time = None
        self.last_evaluation_start_time = None

        # counters
        self.total_steps_counters = {
            Episodes: 0,
            EnvironmentSteps: 0,
            TrainingSteps: 0,
            Frames: 0
        }

        # set self as the parent of all the level managers
        for level_manager in self.level_managers:
            level_manager.parent_block_scheduler = self

        self.block_logger = Logger()

    def setup_logger(self, parent_block_factory: 'BlockFactory') -> None:
        # dump documentation
        logger_prefix = "{block_name}".format(block_name=self.name)
        self.block_logger.set_logger_filenames(self.task_parameters.experiment_path, logger_prefix=logger_prefix,
                                               add_timestamp=True, task_id=self.task_parameters.task_index)
        self.block_logger.dump_documentation(str(parent_block_factory))
        self.parent_block_factory = parent_block_factory
        [manager.as_level_manager.setup_logger() for manager in self.level_managers]

    @property
    def phase(self) -> RunPhase:
        """
        Get the phase of the block scheduler
        :return: the current phase
        """
        return self._phase

    @phase.setter
    def phase(self, val: RunPhase):
        """
        Change the phase of the block scheduler and all the hierarchy levels below it
        :param val: the new phase
        :return: None
        """
        self._phase = val
        for level_manager in self.level_managers:
            level_manager.phase = val
        for environment in self.environments:
            environment.phase = val

    def set_session(self, sess) -> None:
        """
        Set the deep learning framework session for all the modules in the block
        :return: None
        """
        [manager.as_level_manager.set_session(sess) for manager in self.level_managers]

    def heatup(self, steps: PlayingStepsType) -> None:
        """
        Perform heatup for several steps, which means taking random actions and storing the results in memory
        :param steps: the number of steps as a tuple of steps time and steps count
        :return: None
        """
        if steps.num_steps > 0:
            self.phase = RunPhase.HEATUP
            screen.log_title("{}: Starting heatup".format(self.name))
            self.heatup_start_time = time.time()

            # reset all the levels before starting to heatup
            self.reset(force_environment_reset=True)

            # TODO: add a keep running until completing an episode flag
            # act on the environment
            self.act(steps, continue_until_game_over=True)
            # steps_acted, _ = self.act(
            #                   steps.__class__(1),
            #                   continue_until_game_over=True)  # make sure that at least one episode is completed
            #
            # remaining_steps = steps.__class__(steps.num_steps - steps_acted)
            # if remaining_steps.num_steps > 0:
            #     _, done = self.act(remaining_steps)
            #
            #     # end the episode if it was partial
            #     if not done:
            #         self.end_episode()

            # training phase
            self.phase = RunPhase.UNDEFINED

    def end_episode(self) -> None:
        """
        End an episode and reset all the episodic parameters
        :return: None
        """
        self.total_steps_counters[Episodes] += 1

        [manager.end_episode() for manager in self.level_managers]

        self.reset()

    def train(self, steps: TrainingSteps) -> None:
        """
        Perform several training iterations for all the levels in the hierarchy
        :param steps: number of training iterations to perform
        :return: None
        """
        # perform several steps of training interleaved with acting
        count_end = self.total_steps_counters[TrainingSteps] + steps.num_steps
        while self.total_steps_counters[TrainingSteps] < count_end:
            self.total_steps_counters[TrainingSteps] += 1
            # losses = call_method_for_all(list(self.level_managers), 'as_level_manager.train')
            losses = [manager.as_level_manager.train() for manager in self.level_managers]
            # self.loss.add_sample(loss)

    def reset(self, force_environment_reset=False) -> None:
        """
        Reset an episode for all the levels
        :param force_environment_reset: force the environment to reset the episode even if it has some conditions that
                                        tell it not to. for example, if ale life is lost, gym will tell the agent that
                                        the episode is finished but won't actually reset the episode if there are more
                                        lives available
        :return: None
        """

        [manager.reset() for manager in self.level_managers]
        [environment.reset(force_environment_reset) for environment in self.environments]

    def act(self, steps: PlayingStepsType, return_on_game_over: bool=False, continue_until_game_over=False,
            keep_networks_in_sync=False) -> (int, bool):
        """
        Do several steps of acting on the environment
        :param steps: the number of steps as a tuple of steps time and steps count
        :param return_on_game_over: finish acting if an episode is finished
        :param continue_until_game_over: continue playing until an episode was completed
        :param keep_networks_in_sync: sync the network parameters with the global network before each episode
        :return: the actual number of steps done, a boolean value that represent if the episode was done when finishing
                 the function call
        """
        # perform several steps of playing
        result = None

        hold_until_a_full_episode = True if continue_until_game_over else False
        initial_count = self.total_steps_counters[steps.__class__]
        count_end = initial_count + steps.num_steps

        # The assumption here is that the total_steps_counters are each updated when an event
        #  takes place (i.e. an episode ends)
        # TODO - The counter of frames is not updated correctly. need to fix that.
        while self.total_steps_counters[steps.__class__] < count_end or hold_until_a_full_episode:
            result = self.top_level_manager.step(None)
            # result will be None if at least one level_manager decided not to play (= all of his agents did not play)
            # causing the rest of the level_managers down the stack not to play either, and thus the entire block did
            # not acted
            if result is None:
                break

            if self.phase != RunPhase.TEST:
                self.total_steps_counters[EnvironmentSteps] += 1

            if result.game_over:
                hold_until_a_full_episode = False
                self.end_episode()
                if keep_networks_in_sync:
                    self.sync_block()
                if return_on_game_over:
                    return self.total_steps_counters[EnvironmentSteps] - initial_count, True

        # return the game over status
        if result:
            return self.total_steps_counters[EnvironmentSteps] - initial_count, result.game_over
        else:
            return self.total_steps_counters[EnvironmentSteps] - initial_count, False

    def train_and_act(self, steps: StepMethod) -> None:
        """
        Train the agent by doing several acting steps followed by several training steps continually
        :param steps: the number of steps as a tuple of steps time and steps count
        :return: None
        """
        # perform several steps of training interleaved with acting
        count_end = self.total_steps_counters[steps.__class__] + steps.num_steps
        if steps.num_steps > 0:
            self.phase = RunPhase.TRAIN
            self.reset(force_environment_reset=True)
            #TODO - the below while loop should end with full episodes, so to avoid situations where we have partial episodes in memory
            while self.total_steps_counters[steps.__class__] < count_end:
                # The actual steps being done on the environment are decided by the agents themselves.
                # This is just an high-level controller.
                self.act(EnvironmentSteps(1))
                self.train(TrainingSteps(1))
            self.phase = RunPhase.UNDEFINED

    def sync_block(self) -> None:
        """
        Sync the global network parameters to the block
        :return:
        """
        [manager.as_level_manager.sync() for manager in self.level_managers]

    def evaluate(self, steps: PlayingStepsType) -> None:
        """
        Perform evaluation for several steps
        :param steps: the number of steps as a tuple of steps time and steps count
        :return: None
        """
        if steps.num_steps > 0:
            self.phase = RunPhase.TEST
            screen.log_title("{}: Starting evaluation phase".format(self.name))
            self.last_evaluation_start_time = time.time()

            # reset all the levels before starting to evaluate
            self.reset(force_environment_reset=True)
            self.sync_block()

            initial_episode_idx = [env.episode_idx for env in self.environments]
            initial_successes = [env.success_counter for env in self.environments]

            # act on the environment
            _, done = self.act(steps, keep_networks_in_sync=True)

            # end the episode if it was partial
            if not done:
                self.end_episode()

            updated_episode_idx = [env.episode_idx for env in self.environments]
            updated_successes = [env.success_counter for env in self.environments]

            # TODO: keep track of the evaluation episodes and their results
            # TODO: sync networks?

            success_rate = np.squeeze(np.subtract(updated_successes, initial_successes) \
                           / np.subtract(updated_episode_idx, initial_episode_idx))
            screen.log_title("{}: Finished evaluation phase. Successes rate = {}"
                             .format(self.name, success_rate))
            self.phase = RunPhase.UNDEFINED

    def save_checkpoint(self):
        [manager.as_level_manager.save_checkpoint() for manager in self.level_managers]

    def improve(self):
        """
        The main loop of the run.
        Defined in the following steps:
        1. Heatup
        2. Repeat:
            2.1. Repeat:
                2.1.1. Act
                2.1.2. Train
            2.2. Evaluate
            2.3. Save checkpoint
        :return: None
        """

        # initialize the network parameters from the global network
        self.sync_block()

        # heatup
        self.heatup(self.heatup_steps)

        # improve
        if self.task_parameters.task_index is not None:
            screen.log_title("Starting to improve {} task index {}".format(self.name, self.task_parameters.task_index))
        else:
            screen.log_title("Starting to improve {}".format(self.name))
        self.training_start_time = time.time()
        count_end = self.improve_steps.num_steps
        while self.total_steps_counters[self.improve_steps.__class__] < count_end:
            self.train_and_act(self.steps_between_evaluation_periods)
            self.evaluate(self.evaluation_steps)
            self.save_checkpoint()  # TODO: how to control frequency of saving checkpoints


def start_block(block_factory: 'BlockFactory', task_parameters: 'TaskParameters'):
    block_scheduler = block_factory.create_block(task_parameters)
    block_scheduler.setup_logger(block_factory)

    # let the adventure begin
    if task_parameters.evaluate_only:
        block_scheduler.evaluate(EnvironmentSteps(sys.maxsize))
    else:
        block_scheduler.improve()
