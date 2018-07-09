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
import glob
import os
import shutil
import signal
import subprocess
import sys
import time
from os import path
from importlib import import_module

import numpy as np
import pandas as pd

# -*- coding: utf-8 -*-
# import presets
from logger import screen
from utils import list_all_classes_in_module

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preset',
                        help="(string) Name of a preset to run (as configured in presets.py)",
                        default=None,
                        type=str)
    parser.add_argument('-ip', '--ignore_presets',
                        help="(string) Name of a preset(s) to ignore (comma separated, and as configured in presets.py)",
                        default=None,
                        type=str)
    parser.add_argument('-v', '--verbose',
                        help="(flag) display verbose logs in the event of an error",
                        action='store_true')
    parser.add_argument('--stop_after_first_failure',
                        help="(flag) stop executing tests after the first error",
                        action='store_true')

    args = parser.parse_args()
    if args.preset is not None:
        presets_lists = [args.preset]
    else:
        # presets_lists = list_all_classes_in_module(presets)
        presets_lists = [f[:-3] for f in os.listdir('presets') if f[-3:] == '.py' and not f == '__init__.py']
    win_size = 10
    fail_count = 0
    test_count = 0
    read_csv_tries = 50

    # create a clean experiment directory
    test_name = '__test'
    test_path = os.path.join('./experiments', test_name)
    if path.exists(test_path):
        shutil.rmtree(test_path)
    if args.ignore_presets is not None:
        presets_to_ignore = args.ignore_presets.split(',')
    else:
        presets_to_ignore = []
    for idx, preset_name in enumerate(presets_lists):
        if preset_name not in presets_to_ignore:
            try:
                preset = import_module('presets.{}'.format(preset_name))
            except:
                screen.error("Failed to load perset <{}>".format(preset_name), crash=False)
                continue

            test_params = preset.graph_manager.test_params
            if not test_params.test:
                continue
            if args.stop_after_first_failure and fail_count > 0:
                break

            test_count += 1

            # run the experiment in a separate thread
            screen.log_title("Running test {}".format(preset_name))
            log_file_name = 'test_log_{preset_name}.txt'.format(preset_name=preset_name)

            cmd = (
                'CUDA_VISIBLE_DEVICES='' python3 coach.py '
                '-p {preset_name} '
                '-e {test_name} '
                '-n {num_workers} '
                '--seed 0 '
                '-ew '
                '{level} '
                '&> {log_file_name} '
            ).format(
                preset_name=preset_name,
                test_name=test_name,
                num_workers=test_params.num_workers,
                log_file_name=log_file_name,
                level='-lvl ' + test_params.level if test_params.level else ''
            )

            p = subprocess.Popen(cmd, shell=True, executable="/bin/bash", preexec_fn=os.setsid)

            # get the csv with the results
            csv_path = None
            csv_paths = []

            if test_params.num_workers > 1:
                # we have an evaluator
                reward_str = 'Shaped Evaluation Reward'
                filename_pattern = 'worker_{}*.csv'.format(test_params.num_workers)  # TODO: find a better way to extract csv


            else:
                reward_str = 'Training Reward'
                filename_pattern = '*.csv'  # TODO: find a better way to extract csv

            initialization_error = False
            test_passed = False

            tries_counter = 0
            while not csv_paths:
                csv_paths = glob.glob(path.join(test_path, '*', filename_pattern))
                if tries_counter > read_csv_tries:
                    break
                tries_counter += 1
                time.sleep(1)

            if csv_paths:
                csv_path = csv_paths[0]

                # verify results
                csv = None
                time.sleep(1)
                averaged_rewards = [0]

                last_num_episodes = 0
                while csv is None or csv['Episode #'].values[-1] < test_params.max_episodes_to_achieve_reward:
                    try:
                        csv = pd.read_csv(csv_path)
                    except:
                        # sometimes the csv is being written at the same time we are
                        # trying to read it. no problem -> try again
                        continue

                    if reward_str not in csv.keys():
                        continue

                    rewards = csv[reward_str].values
                    rewards = rewards[~np.isnan(rewards)]

                    if len(rewards) >= win_size:
                        averaged_rewards = np.convolve(rewards, np.ones(win_size) / win_size, mode='valid')
                    else:
                        time.sleep(1)
                        continue

                    # print progress
                    percentage = int((100*last_num_episodes)/test_params.max_episodes_to_achieve_reward)
                    sys.stdout.write("\rReward: ({}/{})".format(round(averaged_rewards[-1], 1), test_params.min_reward_threshold))
                    sys.stdout.write(' Episode: ({}/{})'.format(last_num_episodes, test_params.max_episodes_to_achieve_reward))
                    sys.stdout.write(' {}%|{}{}|  '.format(percentage, '#'*int(percentage/10), ' '*(10-int(percentage/10))))
                    sys.stdout.flush()

                    if csv['Episode #'].shape[0] - last_num_episodes <= 0:
                        continue

                    last_num_episodes = csv['Episode #'].values[-1]

                    # check if reward is enough

                    if np.any(averaged_rewards > test_params.min_reward_threshold):
                        test_passed = True
                        break
                    time.sleep(1)

            # kill test and print result
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            if test_passed:
                screen.success("Passed successfully")
            else:
                if csv_paths:
                    screen.error("Failed due to insufficient reward", crash=False)
                    screen.error("test_params.max_episodes_to_achieve_reward: {}".format(test_params.max_episodes_to_achieve_reward), crash=False)
                    screen.error("test_params.min_reward_threshold: {}".format(test_params.min_reward_threshold), crash=False)
                    screen.error("averaged_rewards: {}".format(averaged_rewards), crash=False)
                    screen.error("episode number: {}".format(csv['Episode #'].values[-1]), crash=False)
                else:
                    screen.error("csv file never found", crash=False)
                    if args.verbose:
                        screen.error("command exitcode: {}".format(p.returncode), crash=False)
                        screen.error(open(log_file_name).read(), crash=False)

                fail_count += 1
            shutil.rmtree(test_path)


    screen.separator()
    if fail_count == 0:
        screen.success(" Summary: " + str(test_count) + "/" + str(test_count) + " tests passed successfully")
    else:
        screen.error(" Summary: " + str(test_count - fail_count) + "/" + str(test_count) + " tests passed successfully")
