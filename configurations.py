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


from enum import Enum
import utils
import json
import types
from collections import OrderedDict
from typing import Dict, List, Union
from core_types import GoalTypes, TrainingSteps, EnvironmentSteps
import sys
import ast


# TODO: this function is not working correctly anymore + is duplicated in presets
def json_to_preset(json_path):
    with open(json_path, 'r') as json_file:
        run_dict = json.loads(json_file.read())

    if run_dict['preset'] is None:
        tuning_parameters = Preset(eval(run_dict['agent_type']), eval(run_dict['environment_type']),
                                   eval(run_dict['exploration_policy_type']))
    else:
        tuning_parameters = eval(run_dict['preset'])()
        # Override existing parts of the preset
        if run_dict['agent_type'] is not None:
            tuning_parameters.agent = eval(run_dict['agent_type'])()

        if run_dict['environment_type'] is not None:
            tuning_parameters.env = eval(run_dict['environment_type'])()

        if run_dict['exploration_policy_type'] is not None:
            tuning_parameters.exploration = eval(run_dict['exploration_policy_type'])()

    # human control
    if run_dict['play']:
        tuning_parameters.agent.type = 'HumanAgent'
        tuning_parameters.env.human_control = True
        tuning_parameters.num_heatup_steps = 0

    if run_dict['level']:
        tuning_parameters.env.level = run_dict['level']

    if run_dict['custom_parameter'] is not None:
        unstripped_key_value_pairs = [pair.split('=') for pair in run_dict['custom_parameter'].split(';')]
        stripped_key_value_pairs = [tuple([pair[0].strip(), ast.literal_eval(pair[1].strip())]) for pair in
                                    unstripped_key_value_pairs]

        # load custom parameters into run_dict
        for key, value in stripped_key_value_pairs:
            run_dict[key] = value

    for key in ['agent_type', 'environment_type', 'exploration_policy_type', 'preset', 'custom_parameter']:
        run_dict.pop(key, None)

    # load parameters from run_dict to tuning_parameters
    for key, value in run_dict.items():
        if ((sys.version_info[0] == 2 and type(value) == unicode) or
                (sys.version_info[0] == 3 and type(value) == str)):
            value = '"{}"'.format(value)
        exec('tuning_parameters.{} = {}'.format(key, value)) in globals(), locals()

    return tuning_parameters


class Frameworks(utils.Enum):
    TensorFlow = "TensorFlow"
    Neon = "neon"


class InputTypes(object):
    Observation = "Observation"
    Measurements = "Measurements"
    GoalVector = "GoalVector"
    Action = "Action"
    TimedObservation = "TimedObservation"
    ObservationIdentity = "ObservationIdentity"
    GoalIdentity = "GoalIdentity"


class EmbedderScheme(Enum):
    Empty = "Empty"
    Shallow = "Shallow"
    Medium = "Medium"
    Deep = "Deep"


class OutputTypes(object):
    Q = 'q_head:QHead'
    DuelingQ = 'dueling_q_head:DuelingQHead'
    V = 'v_head:VHead'
    Pi = 'policy_head:PolicyHead'
    MeasurementsPrediction = 'measurements_prediction_head:MeasurementsPredictionHead'
    DNDQ = 'dnd_q_head:DNDQHead'
    NAF = 'naf_head:NAFHead'
    PPO = 'ppo_head:PPOHead'
    PPO_V = 'ppo_v_head:PPOVHead'
    CategoricalQ = 'categorical_q_head:CategoricalQHead'
    QuantileRegressionQ = 'quantile_regression_q_head:QuantileRegressionQHead'
    GoalMapping = 'goal_mapping_head:GoalMappingHead'


class MiddlewareTypes(object):
    LSTM = 'lstm_embedder:LSTM_Embedder'
    FC = 'fc_embedder:FC_Embedder'


def iterable_to_items(obj):
    if isinstance(obj, dict) or isinstance(obj, OrderedDict) or isinstance(obj, types.MappingProxyType):
        items = obj.items()
    elif isinstance(obj, list):
        items = enumerate(obj)
    else:
        raise ValueError("The given object is not a dict or a list")
    return items


def unfold_dict_or_list(obj: Union[Dict, List, OrderedDict]):
    """
    Recursively unfolds all the parameters in dictionaries and lists
    :param obj: a dictionary or list to unfold
    :return: the unfolded parameters dictionary
    """
    parameters = OrderedDict()
    items = iterable_to_items(obj)
    for k, v in items:
        if isinstance(v, dict) or isinstance(v, list) or isinstance(v, OrderedDict):
            if 'tensorflow.' not in str(v.__class__):
                parameters[k] = unfold_dict_or_list(v)
        elif 'tensorflow.' in str(v.__class__):
            parameters[k] = v
        elif hasattr(v, '__dict__'):
            sub_params = v.__dict__
            if '__objclass__' not in sub_params.keys():
                try:
                    parameters[k] = unfold_dict_or_list(sub_params)
                except RecursionError:
                    parameters[k] = sub_params
                parameters[k]['__class__'] = v.__class__.__name__
            else:
                # unfolding this type of object will result in infinite recursion
                parameters[k] = sub_params
        else:
            parameters[k] = v
    if not isinstance(obj, OrderedDict) and not isinstance(obj, list):
        parameters = OrderedDict(sorted(parameters.items()))
    return parameters


class Parameters(object):
    def __setattr__(self, key, value):
        caller_name = sys._getframe(1).f_code.co_name

        if caller_name != '__init__' and not hasattr(self, key):
            raise TypeError("Parameter '{}' does not exist in {}. Parameters are only to be defined in a constructor of"
                            " a class inheriting from Parameters. In order to explicitly register a new parameter "
                            "outside of a constructor use register_var().".
                            format(key, self.__class__))
        object.__setattr__(self, key, value)

    def register_var(self, key, value):
        if hasattr(self, key):
            raise TypeError("Cannot register an already existing parameter '{}'. ".format(key))
        object.__setattr__(self, key, value)

    def __str__(self):
        result = "\"{}\" {}\n".format(self.__class__.__name__,
                                   json.dumps(unfold_dict_or_list(self.__dict__), indent=4, default=repr))
        return result


class AlgorithmParameters(Parameters):
    def __init__(self):
        # Architecture parameters
        self.use_accumulated_reward_as_measurement = False
        self.add_a_normalized_timestep_to_the_observation = False

        # Agent parameters
        self.num_consecutive_playing_steps = EnvironmentSteps(1)
        self.num_consecutive_training_steps = 1  # TODO: update this to TrainingSteps

        self.heatup_using_network_decisions = False
        self.discount = 0.99
        self.apply_gradients_every_x_episodes = 5
        self.num_steps_between_copying_online_weights_to_target = TrainingSteps(1000)
        self.rate_for_copying_weights_to_target = 1.0
        # self.targets_horizon = 'N-Step'
        self.load_memory_from_file_path = None
        self.collect_new_data = True

        # HRL / HER related params
        # self.add_intrinsic_reward_for_reaching_the_goal = False
        # self.state_value_to_use_as_goal = GoalTypes.Measurements
        # self.distance_from_goal_threshold = 0.1
        # self.ignore_extrinsic_reward = False
        # self.subagent_timestep_limit = 2
        # self.share_input_embedder_between_subagents = False
        # self.goal_pooling = 1  # this is used in FeUdal Network

        # distributed agents params
        self.shared_optimizer = True
        self.share_statistics_between_workers = True

        # intrinsic reward
        self.scale_external_reward_by_intrinsic_reward_value = False


class GeneralParameters(Parameters):
    def __init__(self):
        super().__init__()

        # setting a seed will only work for non-parallel algorithms. Parallel algorithms add uncontrollable noise in
        # the form of different workers starting at different times, and getting different assignments of CPU
        # time from the OS.

        # Testing parameters
        self.test = False
        self.test_min_return_threshold = 0
        self.test_max_step_threshold = 1
        self.test_num_workers = 1


class NetworkParameters(Parameters):
    def __init__(self):
        super().__init__()
        self.framework = Frameworks.TensorFlow
        self.sess = None

        # hardware parameters
        self.force_cpu = False

        # distributed training options
        self.num_threads = 1
        self.synchronize_over_num_threads = 1
        self.distributed = False
        self.async_training = False
        self.shared_optimizer = True

        # regularization
        self.clip_gradients = None
        self.kl_divergence_constraint = None
        self.l2_regularization = 0

        # checkpoints
        self.save_model_sec = None
        self.save_model_dir = None
        self.checkpoint_restore_dir = None

        # learning rate
        self.learning_rate = 0.00025
        self.learning_rate_decay_rate = 0
        self.learning_rate_decay_steps = 0

        # structure
        self.input_types = []
        self.middleware_type = None
        self.output_types = []
        self.num_output_head_copies = 1
        self.loss_weights = []
        self.rescale_gradient_from_head_by_factor = [1]
        self.use_separate_networks_per_head = False
        self.middleware_hidden_layer_size = 512
        self.hidden_layers_activation_function = 'relu'
        self.optimizer_type = 'Adam'
        self.optimizer_epsilon = 0.0001
        self.batch_size = 32
        self.replace_mse_with_huber_loss = False
        # TODO:should this be 255 even for vector observations?
        self.input_rescaler = 255.0  # TODO: considering removing this in favor of filters
        self.create_target_network = False

        # Framework support
        self.neon_support = False
        self.tensorflow_support = True


class InputEmbedderParameters(Parameters):
    def __init__(self):
        super().__init__()
        self.activation_function = 'relu'
        self.embedder_scheme = EmbedderScheme.Medium
        self.embedder_width_multiplier = 1
        self.use_batchnorm = False
        self.use_dropout = False
        self.input_rescaler = 255.0
        self.name = "embedder"

    @property
    def path(self):
        return {
            "image": 'image_embedder:ImageEmbedder',
            "vector": 'vector_embedder:VectorEmbedder'
        }


class VisualizationParameters(Parameters):
    def __init__(self):
        super().__init__()
        # Visualization parameters
        self.print_summary = True
        self.dump_csv = True
        self.dump_gifs = False
        self.dump_mp4 = False
        self.dump_signals_to_csv_every_x_episodes = 5
        self.dump_in_episode_signals = False
        self.render = False
        self.native_rendering = False
        self.max_fps_for_human_control = 10
        self.tensorboard = False
        self.video_dump_methods = []  # a list of dump methods which will be checked one after the other until the first
                                      # dump method that returns false for should_dump()
        self.add_rendered_image_to_env_response = False


class Human(AlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.type = 'HumanAgent'
        self.num_episodes_in_experience_replay = 10000000


# TODO: where to put this?
# class DuelingDQN(DQN):
#     def __init__(self):
#         super().__init__()
#         self.type = 'DQNAgent'
#         self.output_types = [OutputTypes.DuelingQ]


class DDDPG(AlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.type = 'DDPGAgent'
        self.input_types = [InputEmbedderParameters(), InputEmbedderParameters()]
        self.output_types = [OutputTypes.V]  # V is used because we only want a single Q value
        self.loss_weights = [1.0]
        self.hidden_layers_activation_function = 'relu'
        self.num_episodes_in_experience_replay = 10000
        self.num_steps_between_copying_online_weights_to_target = 10
        self.rate_for_copying_weights_to_target = 1
        self.shared_optimizer = True
        self.async_training = True


class BC(AlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.type = 'BCAgent'
        self.input_types = [InputEmbedderParameters()]
        self.output_types = [OutputTypes.Q]
        self.loss_weights = [1.0]
        self.collect_new_data = False
        self.evaluate_every_x_training_iterations = 50000


# from exploration_policies.exploration_policy import ExplorationParameters
# from memories.memory import MemoryParameters


class AgentParameters(GeneralParameters):
    def __init__(self, algorithm: AlgorithmParameters, exploration: 'ExplorationParameters', memory: 'MemoryParameters',
                 networks: Dict[str, NetworkParameters], visualization: VisualizationParameters=VisualizationParameters()):
        """
        :param algorithm: the algorithmic parameters
        :param exploration: the exploration policy parameters
        :param memory: the memory module parameters
        :param networks: the parameters for the networks of the agent
        :param visualization: the visualization parameters
        """
        super().__init__()
        self.visualization = visualization
        self.algorithm = algorithm
        self.exploration = exploration
        self.memory = memory
        self.network_wrappers = networks
        self.input_filter = None
        self.output_filter = None
        self.full_name_id = None  # TODO: do we really want to hold this parameters here?
        self.name = None
        self.task_parameters = None

    @property
    def path(self):
        return 'agents.agent:Agent'

