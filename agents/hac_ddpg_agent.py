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

import copy
from typing import Union

import numpy as np

from agents.actor_critic_agent import ActorCriticAgent
from agents.agent import Agent
from agents.ddpg_agent import DDPGAgent, DDPGAgentParameters, DDPGAlgorithmParameters
from architectures.tensorflow_components.heads.policy_head import PolicyHeadParameters
from architectures.tensorflow_components.heads.v_head import VHeadParameters
from architectures.tensorflow_components.middlewares.fc_middleware import FCMiddlewareParameters
from base_parameters import NetworkParameters, AlgorithmParameters, \
    AgentParameters, InputEmbedderParameters, EmbedderScheme
from core_types import ActionInfo, EnvironmentSteps, EnvResponse, RunPhase
from exploration_policies.ou_process import OUProcessParameters
from memories.episodic_experience_replay import EpisodicExperienceReplayParameters
from spaces import BoxActionSpace


class HACDDPGAlgorithmParameters(DDPGAlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.sub_goal_testing_rate = 0.5
        self.time_limit = 40


class HACDDPGAgentParameters(DDPGAgentParameters):
    def __init__(self):
        super().__init__()
        self.algorithm = HACDDPGAlgorithmParameters()

    @property
    def path(self):
        return 'agents.hac_ddpg_agent:HACDDPGAgent'


# Hierarchical Actor Critic Subgoal DDPG Agent - https://arxiv.org/pdf/1712.00948.pdf
class HACDDPGAgent(DDPGAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.sub_goal_testing_episode = False

    def handle_episode_ended(self) -> None:
        super().handle_episode_ended()
        if self.phase == RunPhase.TRAIN:
            self.sub_goal_testing_episode = np.random.rand() < self.ap.algorithm.sub_goal_testing_rate
            if self.sub_goal_testing_episode:
                self.exploration_policy.change_phase(RunPhase.TEST)
            else:
                self.exploration_policy.change_phase(self.phase)

    def update_transition_before_adding_to_replay_buffer(self, transition):
        if self.sub_goal_testing_episode:
            if self.current_episode_steps_counter >= self.ap.algorithm.time_limit - 1 and transition.reward < 0:
                transition.reward = -self.ap.algorithm.time_limit
        return transition
