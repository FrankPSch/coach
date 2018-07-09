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
from typing import Tuple, List

from agents.composite_agent import CompositeAgent, SingleDecider
from graph_managers.graph_manager import GraphManager, ScheduleParameters
from base_parameters import AgentParameters, VisualizationParameters, TaskParameters
from environments.environment import EnvironmentParameters, Environment
from level_manager import LevelManager
from utils import short_dynamic_import


class CustomActorCriticGraphManager(GraphManager):
    """
    A custom actor critic graph manager creates a single composite agent and a single environment.
    The composite agent contains a single actor agent which gets to decide on the composite agent
    actions, and one or several critic agents, which are only observing the results and criticizing the actor.
    """
    def __init__(self, actor_params: AgentParameters, critic_params: AgentParameters,
                 schedule_params: ScheduleParameters, env_params: EnvironmentParameters,
                 vis_params: VisualizationParameters = VisualizationParameters()):
        super().__init__('custom_actor_critic_graph', schedule_params, vis_params)
        self.actor_params = actor_params
        self.critic_params = critic_params
        self.env_params = env_params

    def _create_graph(self, task_parameters: TaskParameters) -> Tuple[List[LevelManager], List[Environment]]:
        env = short_dynamic_import(self.env_params.path)(**self.env_params.__dict__,
                                                         visualization_parameters=self.visualization_parameters)
        self.actor_params.task_parameters = task_parameters  # TODO: this should probably be passed in a different way
        self.critic_params.task_parameters = task_parameters  # TODO: this should probably be passed in a different way

        composite_agent = CompositeAgent(
            agents_parameters={
                "actor": self.actor_params,
                "critic": self.critic_params
            },
            visualization_parameters=self.visualization_parameters,
            raw_observation_space=env.state_space['observation'],
            out_action_space=env.action_space,
            decision_makers={
                "actor": True,
                "critic": False
            },
            decision_policy=SingleDecider(default_decision_maker="actor"),
            # Note that in this case only the input filter from the actor is taken into account.
            # input_filter=self.actor_params.input_filter,  # TODO: ask gal about removing this
            # output_filter=self.actor_params.output_filter,
            name="actor_critic",
        )
        level_manager = LevelManager(
            agents=composite_agent,
            environment=env,
            name="main_level",
        )

        return [level_manager], [env]