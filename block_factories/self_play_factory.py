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

from block_factories.block_factory import BlockFactory
from block_scheduler import BlockScheduler
from configurations import AgentParameters, EnvironmentParameters, VisualizationParameters
from utils import short_dynamic_import
from agents.composite_agent import CompositeAgent, SingleDecider
from level_manager import LevelManager
from core_types import Episodes, EnvironmentSteps, TrainingSteps


class SelfPlayFactory(BlockFactory):
    """
    A block factory is responsible for creating and initializing a block, including all its internal components.
    A simple HRL factory creates a deep hierarchy with a single composite agent per hierarchy level, and a single
    environment which is interacted with.
    """
    def __init__(self, agent_params: AgentParameters, env_params: EnvironmentParameters,
                 vis_params: VisualizationParameters):
        super().__init__()
        self.agent_params = agent_params
        self.env_params = env_params
        self.vis_params = vis_params

    def _create_block(self, task_index: int, device=None) -> BlockScheduler:
        """
        Create all the block modules and the block scheduler
        :param task_index: the index of the process on which the worker will be run
        :return: the initialized block scheduler
        """
        raise NotImplementedError("")
