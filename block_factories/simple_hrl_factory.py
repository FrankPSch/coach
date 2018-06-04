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

from block_factories.block_factory import BlockFactory, TaskParameters
from block_scheduler import BlockScheduler, BlockSchedulerParameters
from configurations import AgentParameters, VisualizationParameters
from environments.environment import EnvironmentParameters
from utils import short_dynamic_import
from agents.composite_agent import CompositeAgent, SingleDecider
from level_manager import LevelManager
from typing import List
from utils import set_member_values_for_all
from spaces import Attention, GoalTypes
import numpy as np


class SimpleHRLFactory(BlockFactory):
    """
    A block factory is responsible for creating and initializing a block, including all its internal components.
    A simple HRL factory creates a deep hierarchy with a single composite agent per hierarchy level, and a single
    environment which is interacted with.
    """
    def __init__(self, agents_params: List[AgentParameters], env_params: EnvironmentParameters,
                 schedule_params: BlockSchedulerParameters, vis_params: VisualizationParameters):
        """
        :param agents_params: the parameters of all the agents in the hierarchy starting from the top level of the
                              hierarchy to the bottom level
        :param env_params: the parameters of the environment
        :param vis_params:
        """
        super().__init__()
        self.agents_params = agents_params
        self.env_params = env_params
        self.schedule_params = schedule_params
        self.vis_params = vis_params

        if len(self.agents_params) < 2:
            raise ValueError("The HRL factory must receive the agent parameters for at least two levels of the "
                             "hierarchy. Otherwise, use the basic RL factory.")

    def _create_block(self, task_parameters: TaskParameters) -> BlockScheduler:
        env = short_dynamic_import(self.env_params.path)(**self.env_params.__dict__,
                                                         visualization_parameters=self.vis_params)

        for agent_params in self.agents_params:
            agent_params.task_parameters = task_parameters

        # we need to build the hierarchy in reverse order (from the bottom up) in order for the spaces of each level
        # to be known
        level_managers = []
        current_env = env
        out_action_space = env.action_space
        for level_idx, agent_params in reversed(list(enumerate(self.agents_params))):
            # in action space
            if level_idx == 0:
                in_action_space = None
            else:
                attention_size = (env.state_space['observation'].shape - 1)//4
                in_action_space = Attention(shape=2, low=0, high=env.state_space['observation'].shape - 1,
                                            forced_attention_size=attention_size)  # TODO: pass the size somehow
                agent_params.output_filter.action_filters['masking'].set_masking(0, attention_size)

            agent_params.visualization = self.vis_params
            composite_agent = CompositeAgent(
                agents_parameters={
                    "agent": agent_params,
                },
                visualization_parameters=self.vis_params,
                raw_observation_space=env.state_space['observation'],  # TODO: add this to the interface?
                in_action_space=in_action_space,  # TODO: it should be possible to define this per preset
                out_action_space=out_action_space,
                decision_makers={
                    "agent": True,
                },
                decision_policy=SingleDecider(default_decision_maker="agent"),
                name="simple_hrl_agent_level_{}".format(level_idx)
            )
            level_manager = LevelManager(
                agents=composite_agent,
                environment=current_env,
                real_environment=env,
                name="level_{}".format(level_idx)
            )
            current_env = level_manager
            level_managers.insert(0, level_manager)

            out_action_space = in_action_space

        block_scheduler = BlockScheduler(
            name='simple_hrl_block',
            level_managers=level_managers,
            environments=[env],
            heatup_steps=self.schedule_params.heatup_steps,
            evaluation_steps=self.schedule_params.evaluation_steps,
            steps_between_evaluation_periods=self.schedule_params.steps_between_evaluation_periods,
            improve_steps=self.schedule_params.improve_steps,
            task_parameters=task_parameters
        )
        return block_scheduler

