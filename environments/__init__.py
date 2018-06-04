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

from utils import Enum
# TODO: do we really need this? currently it's just for warning about uninstalled environments
# from environments.gym_environment import GymEnvironment
# from environments.doom_environment import DoomEnvironment
# from environments.carla_environment import CarlaEnvironment


class EnvTypes(Enum):
    Doom = "DoomEnvironment"
    Gym = "GymEnvironment"
    Carla = "CarlaEnvironment"


def create_environment(tuning_parameters):
    env_type_name, env_type = EnvTypes().verify(tuning_parameters.env.type)
    env = eval(env_type)(tuning_parameters)
    return env

