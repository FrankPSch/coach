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

from environments.environment import Environment
from environments.environment_interface import EnvironmentInterface
from agents.composite_agent import CompositeAgent
from typing import Union, Dict, Tuple, Type
from core_types import EnvResponse, ActionInfo, RunPhase, ActionType
from spaces import ActionSpace, SpacesDefinition


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

    @staticmethod
    def observe(observing_agent: Union['Agent', CompositeAgent], env_response: EnvResponse) -> bool:
        """
        Pass the last response from the environment to the current observing agent
        :return: a done flag
        """
        done = observing_agent.observe(env_response)
        return done

    def step(self) -> Tuple[bool, EnvResponse]:
        """
        Make a single step by getting an action from the current acting agent, stepping the environment and
        returning the response to the current observing agent
        :return: a done flag
        """
        acting_agent = self.next_agent_to_act()

        # get latest observation from environment
        env_response = self.real_environment.last_env_response

        # let the agent observe the result and decide if it wants to terminate the episode
        if env_response is not None:
            done = self.observe(acting_agent, env_response)

        if not done:
            # get action
            action_info = acting_agent.act()

            # continue if at lease one agent in this composite_agent decided to play, and have an action to apply
            if action_info is not None:
                # step environment
                env_response = self.environment.step(action_info.action)

                # if the environment terminated the episode -> let the agent observe the last response
                env_response = self.real_environment.last_env_response  # TODO: make sure we don't miss any information from the lower level manager
                if env_response.game_over:
                    done = self.observe(acting_agent, env_response)  # TODO: acting agent?
            else:  # no agent stepped the environment
                return None

        env_response.game_over = done
        return env_response


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
    # TODO - decide if the new_state in EnvResponse coming back from the environment is the one to be returned to an agent taking an action
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
                 level_behavior_scheme: Type[LevelBehaviorScheme]=SimpleRLScheme
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
        """
        super().__init__()
        if not isinstance(agents, dict):
            # insert the single composite agent to a dictionary for compatibility
            agents = {agents.name: agents}
        if real_environment is None:
            self._real_environment = real_environment = environment
        self.agents = agents

        self.name = name

        # set self as the parent for all the composite agents
        for agent in self.agents.values():
            agent.parent = self
            agent.parent_level_manager = self

        # create all agents in all composite_agents - we do it here so agents will have access to their level manager
        for agent in self.agents.values():
            if isinstance(agent, CompositeAgent):
                agent.create_agents()

        self.as_level_manager = LevelManager.LevelManagerLogic(
            agents=self.agents,
            environment=environment,
            real_environment=real_environment,
            level_behavior_scheme=level_behavior_scheme(self.agents, environment, real_environment),
            name=name)

        self._last_env_response = None
        self.parent_block_scheduler = None

    @property
    def phase(self) -> RunPhase:
        """
        Get the action space of the environment
        :return: the action space
        """
        return self.as_level_manager.phase

    @phase.setter
    def phase(self, val: RunPhase):
        """
        Set the action space of the environment
        :return: None
        """
        self.as_level_manager.phase = val

    def step(self, action: Union[None, Dict[str, ActionType]]) -> EnvResponse:
        """
        Make a single step in the environment using the given action
        :param action: an action to apply to this level manager
        :return: the environment response as returned in get_last_env_response
        """
        return self.as_level_manager.step(action)

    def end_episode(self) -> None:
        """
        End the environment episode
        :return: None
        """
        [agent.end_episode() for agent in self.agents.values()]

    def reset(self, force_environment_reset: bool = False) -> EnvResponse:
        """
        Reset the environment episode parameters
        :param force_environment_reset: in some cases, resetting the environment can be suppressed by the environment
                                        itself. This flag allows force the reset.
        :return: the environment response as returned in get_last_env_response
        """
        [agent.reset() for agent in self.agents.values()]
        return self._last_env_response

    @property
    def action_space(self) -> Dict[str, ActionSpace]:
        """
        Get the action space of each of the agents wrapped in this environment.
        :return: the action space
        """
        cagents_dict = self.as_level_manager.agents
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

    class LevelManagerLogic:
        """
        An helper class for having a different namespace for LevelManager related methods.
        """
        def __init__(self,
                     agents: Dict[str, Union[CompositeAgent, 'Agent']],
                     environment: Union['LevelManager', Environment],
                     real_environment: Environment,
                     level_behavior_scheme: LevelBehaviorScheme,
                     name: str=""
                     ):
            """
            :param agents: a list of agents or composite agents to control
            :param environment: an environment or level manager to control
            :param real_environment: the real environment that is is acted upon. if this is None, it will be replaced by
                                 the environment parameter
            :param level_behavior_scheme: the behavioral scheme that we want this level to follow, while stepping
             through it.
            :param name: the level's name
            """
            self.agents = agents
            self._phase = RunPhase.HEATUP
            self.environment = environment
            self.level_behavior_scheme = level_behavior_scheme
            self.name = name
            self.real_environment = real_environment
            self.build()

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
            :param action: the action to apply to the CompositeAgents held in this level, before beginning following
                           the scheme.
            :return: None
            """
            if action is not None:
                for cagent_name, cagent in self.agents.items():
                    cagent.set_incoming_directive(action) # [cagent_name])  # TODO: we should pass dict from above?

            # TODO: step until episode ended or until something else happens or just one step?
            env_response = self.level_behavior_scheme.step()

            return env_response

        def save_checkpoint(self):
            """
            Save checkpoints of the networks of all agents
            :return: None
            """
            # TODO: should this be done in this manner?
            [agent.save_checkpoint() for agent in self.agents.values()]

        def sync(self) -> None:
            """
            Sync the global network parameters to the block
            :return:
            """
            [agent.sync() for agent in self.agents.values()]
