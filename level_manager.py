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
from typing import Union, Dict, Tuple, Type

from agents.composite_agent import CompositeAgent
from core_types import EnvResponse, ActionInfo, RunPhase, ActionType, StepMethod, EnvironmentSteps
from environments.environment import Environment
from environments.environment_interface import EnvironmentInterface
from spaces import ActionSpace, SpacesDefinition



# TODO: the schemes are not currently being used. We need to rethink this concept since it adds a lot of complexity
class LevelBehaviorScheme(object):
    """
    The level behavior scheme is responsible for defining the internal coordination between the modules of the level.
    Essentially, it should communicate the state of the environment to the appropriate agent, get the next action
    to perform from the appropriate agent, and perform it on the environment.
    """
    def __init__(self,
                 agents: Dict[str, Union['Agent', CompositeAgent]],
                 environment: Union['LevelManager', Environment],
                 real_environment: Environment):
        self.agents = agents
        self.environment = environment
        self.real_environment = real_environment
        self.agent_keys = list(self.agents.keys())
        self.agent_turn = 0

    def next_agent_to_act(self) -> Union['Agent', CompositeAgent]:
        raise NotImplementedError("")

    def move_to_next_agent(self):
        self.agent_turn = (self.agent_turn + 1) % len(self.agent_keys)

    def reset_agents(self):
        for agent in self.agents.values():
            agent.end_episode()
            agent.reset_internal_state()

    def step(self, last_env_response: EnvResponse) -> Tuple[bool, EnvResponse]:
        """
        Make a single step by getting an action from the current acting agent, stepping the environment and
        returning the response to the current observing agent
        :param last_env_response: the last env response to act on
        :return: a done flag and the environment response
        """
        acting_agent = self.next_agent_to_act()

        # get latest environment response
        env_response = last_env_response

        # let the agent observe the result and decide if it wants to terminate the episode
        if env_response is not None:
            done = acting_agent.observe(env_response)

        if not done:
            # get action
            action_info = acting_agent.act()

            # step environment
            env_response = self.environment.step(action_info.action)

            # if the environment terminated the episode -> let the agent observe the last response
            if env_response.game_over:
                done = acting_agent.observe(env_response)  # TODO: acting agent?

        env_response = copy.copy(env_response)

        return done, env_response


class SimpleRLScheme(LevelBehaviorScheme):
    """
    The simple RL scheme defines a behavior where there is a single agent that is acting on the environment
    """
    def __init__(self,
                 agents: Dict[str, Union['Agent', CompositeAgent]],
                 environment: Union['LevelManager', Environment],
                 real_environment: Environment):
        if len(agents) != 1:
            raise ValueError("In a simple RL level behavioral scheme, only a single agent or composite agent is "
                             "assumed.")
        super().__init__(agents, environment, real_environment)

    def next_agent_to_act(self) -> Union['Agent', CompositeAgent]:
        return self.agents[self.agent_keys[0]]


class SelfPlayScheme(LevelBehaviorScheme):
    """
    The self play scheme defines a behavior where there are 2 agents and they are acting in an interleaved fashion
    """
    # TODO - decide if the next_state in EnvResponse coming back from the environment is the one to be returned to an agent taking an action
    #        or alternatively an agent needs to get the new state seen after the other agent(s) has/have taken an action
    #        and is now again the turn of the orignal agent taking an action

    def __init__(self,
                 agents: Dict[str, Union['Agent', CompositeAgent]],
                 environment: Union['LevelManager', Environment],
                 real_environment: Environment):
        super().__init__(agents, environment, real_environment)
        self.agent_turn = 0

    def next_agent_to_act(self):
        agent = self.agents[self.agent_keys[self.agent_turn]]
        return agent


class RealTimeScheme(LevelBehaviorScheme):
    """
    The real time scheme defines a behavior where the next agent that has an action ready is the one that acts on the
    environment. Faster agents can do more steps than slower ones.
    """
    def __init__(self,
                 agents: Dict[str, Union['Agent', CompositeAgent]],
                 environment: Union['LevelManager', Environment],
                 real_environment: Environment):
        super().__init__(agents, environment, real_environment)

    #TODO - fill me


class LevelManager(EnvironmentInterface):
    """
    The LevelManager is in charge of managing a level in the hierarchy of control. Each level can have one or more
    CompositeAgents and an environment to control. Its API is double-folded:
        1. Expose services of a LevelManager such as training the level, or stepping it (while behaving according to a
           LevelBehaviorScheme, e.g. as SelfPlay between two identical agents). These methods are implemented in the
           LevelManagerLogic class.
        2. Disguise as appearing as an environment to the upper level control so it will believe it is interacting with
           an environment. This includes stepping through what appears to be a regular environment, setting its phase
           or resetting it. These methods are implemented directly in LevelManager as it inherits from
           EnvironmentInterface.
    """
    def __init__(self,
                 name: str,
                 agents: Union['Agent', CompositeAgent, Dict[str, Union['Agent', CompositeAgent]]],
                 environment: Union['LevelManager', Environment],
                 real_environment: Environment=None,
                 level_behavior_scheme: Type[LevelBehaviorScheme]=SimpleRLScheme,
                 steps_limit: EnvironmentSteps=EnvironmentSteps(1),
                 reset_after_every_acting_phase: bool=False
                 ):
        """
        A level manager controls a single or multiple composite agents and a single environment.
        The environment can be either a real environment or another level manager behaving as an environment.
        :param agents: a list of agents or composite agents to control
        :param environment: an environment or level manager to control
        :param real_environment: the real environment that is is acted upon. if this is None (which it should be for
         the most bottom level), it will be replaced by the environment parameter. For simple RL schemes, where there
         is only a single level of hierarchy, this removes the requirement of defining both the environment and the
         real environment, as they are the same.
        :param level_behavior_scheme: the behavioral scheme that we want this level to follow, while stepping
             through it.
        :param steps_limit: the number of time steps to run when stepping the internal components
        :param reset_after_every_acting_phase: reset the agent after stepping for steps_limit
        :param name: the level's name
        """
        super().__init__()

        if not isinstance(agents, dict):
            # insert the single composite agent to a dictionary for compatibility
            agents = {agents.name: agents}
        if real_environment is None:
            self._real_environment = real_environment = environment
        self.agents = agents
        self.environment = environment
        self.real_environment = real_environment
        self.level_behavior_scheme = level_behavior_scheme(self.agents, environment, real_environment)
        self.steps_limit = steps_limit
        self.reset_after_every_acting_phase = reset_after_every_acting_phase
        self.full_name_id = self.name = name
        self._phase = RunPhase.HEATUP
        self.level_was_reset = True

        # set self as the parent for all the composite agents
        for agent in self.agents.values():
            agent.parent = self
            agent.parent_level_manager = self

        # create all agents in all composite_agents - we do it here so agents will have access to their level manager
        for agent in self.agents.values():
            if isinstance(agent, CompositeAgent):
                agent.create_agents()

        if not isinstance(self.steps_limit, EnvironmentSteps):
            raise ValueError("The num consecutive steps for acting must be defined in terms of environment steps")
        self.build()

        self._last_env_response = None
        self.parent_graph_manager = None

    def handle_episode_ended(self) -> None:
        """
        End the environment episode
        :return: None
        """
        [agent.handle_episode_ended() for agent in self.agents.values()]

    def reset_internal_state(self, force_environment_reset: bool = False) -> EnvResponse:
        """
        Reset the environment episode parameters
        :param force_environment_reset: in some cases, resetting the environment can be suppressed by the environment
                                        itself. This flag allows force the reset.
        :return: the environment response as returned in get_last_env_response
        """
        [agent.reset_internal_state() for agent in self.agents.values()]
        # TODO - why does this not end the episode for the environment as well?
        self.level_was_reset = True
        return self._last_env_response

    @property
    def action_space(self) -> Dict[str, ActionSpace]:
        """
        Get the action space of each of the agents wrapped in this environment.
        :return: the action space
        """
        cagents_dict = self.agents
        cagents_names = cagents_dict.keys()

        return {name: cagents_dict[name].in_action_space for name in cagents_names}

    def get_random_action(self) -> Dict[str, ActionType]:
        """
        Get a random action from the environment action space
        :return: An action that follows the definition of the action space.
        """
        action_spaces = self.action_space  # The action spaces of the abstracted composite agents in this level
        return {name: action_space.sample() for name, action_space in action_spaces.items()}

    def get_random_action_with_info(self) -> Dict[str, ActionInfo]:
        """
        Get a random action from the environment action space and wrap it with additional info
        :return: An action that follows the definition of the action space with additional generated info.
        """
        return {k: ActionInfo(v) for k, v in self.get_random_action().items()}

    def build(self) -> None:
        """
        Build all the internal components of the level manager (composite agents and environment).
        :return: None
        """
        # TODO: add goal space
        # TODO: move the spaces definition class to the environment?
        action_space = self.environment.action_space
        if isinstance(action_space, dict): # TODO: shouldn't be a dict
            action_space = list(action_space.values())[0]
        spaces = SpacesDefinition(state=self.real_environment.state_space,
                                  goal=self.real_environment.goal_space,  # TODO: what if the agent defines this?
                                  action=action_space,
                                  reward=self.real_environment.reward_space)
        [agent.set_environment_parameters(spaces) for agent in self.agents.values()]

    def setup_logger(self) -> None:
        """
        Setup the logger for all the agents in the level
        :return: None
        """
        [agent.setup_logger() for agent in self.agents.values()]

    def set_session(self, sess) -> None:
        """
        Set the deep learning framework session for all the composite agents in the level manager
        :return: None
        """
        [agent.set_session(sess) for agent in self.agents.values()]

    def train(self) -> None:
        """
        Make a training step for all the composite agents in this level manager
        :return: the loss?
        """
        # result = call_method_for_all(list(self.composite_agents.values()), 'train')
        result = [agent.train() for agent in self.agents.values()]

        # TODO: what to do with the result (=losses)?

    @property
    def phase(self) -> RunPhase:
        """
        Get the phase of the level manager
        :return: the current phase
        """
        return self._phase

    @phase.setter
    def phase(self, val: RunPhase):
        """
        Change the phase of the level manager and all the hierarchy levels below it
        :param val: the new phase
        :return: None
        """
        self._phase = val
        for agent in self.agents.values():
            agent.phase = val

    def step(self, action: Union[None, Dict[str, ActionType]]) -> EnvResponse:
        """
        Run a single step of following the behavioral scheme set for this environment.
        :param action: the action to apply to the agents held in this level, before beginning following
                       the scheme.
        :return: None
        """
        # set the incoming directive for the sub-agent (goal / skill selection / etc.)
        if action is not None:
            for agent_name, agent in self.agents.items():
                agent.set_incoming_directive(action)

        # get last response or initial response from the environment
        env_response = copy.copy(self.real_environment.last_env_response)

        # step for several time steps
        accumulated_reward = 0
        acting_agent = list(self.agents.values())[0]
        done = False
        for i in range(self.steps_limit.num_steps):
            # let the agent observe the result and decide if it wants to terminate the episode
            done = acting_agent.observe(env_response)

            if not done:
                # get action
                action_info = acting_agent.act()

                # step environment
                env_response = self.environment.step(action_info.action)

                # accumulate rewards such that the master policy will see the total reward during the step phase
                accumulated_reward += env_response.reward
            else:
                break

        # if the environment terminated the episode -> let the agent observe the last response
        if self.reset_after_every_acting_phase or env_response.game_over:
            if not done:
                acting_agent.observe(env_response)  # TODO: acting agent?
            self.handle_episode_ended()
            self.reset_internal_state()

        # update the env response that will be exposed to the master agent
        env_response = copy.copy(env_response)
        env_response.reward = accumulated_reward

        return env_response

    def save_checkpoint(self, checkpoint_id: int) -> None:
        """
        Save checkpoints of the networks of all agents
        :return: None
        """
        [agent.save_checkpoint(checkpoint_id) for agent in self.agents.values()]

    def sync(self) -> None:
        """
        Sync the networks of the agents with the global network parameters
        :return:
        """
        [agent.sync() for agent in self.agents.values()]
