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

from filters.observation.observation_filter import ObservationFilter
from spaces import ObservationSpace
from core_types import ObservationType
import numpy as np


class ObservationCropFilter(ObservationFilter):
    """
    Crops the current state observation to a given shape
    """
    def __init__(self, crop_low: np.ndarray=None, crop_high: np.ndarray=None):
        """
        :param crop_low: a vector where each dimension describes the start index for cropping the observation in the
                         corresponding dimension
        :param crop_high: a vector where each dimension describes the end index for cropping the observation in the
                          corresponding dimension
        """
        super().__init__()
        if crop_low is None and crop_high is None:
            raise ValueError("At least one of crop_low and crop_high should be set to a real value. ")
        if crop_low is None:
            crop_low = np.array([None] * len(crop_high))
        if crop_high is None:
            crop_high = np.array([None] * len(crop_low))

        self.crop_low = crop_low
        self.crop_high = crop_high

        if np.any(crop_high < crop_low):
            raise ValueError("Some of the cropping low values are higher than cropping high values")
        if np.any(crop_high) < 0 or np.any(crop_low) < 0:
            raise ValueError("Cropping values cannot be negative")
        if crop_low.shape != crop_high.shape:
            raise ValueError("The low values and high values for cropping must have the same number of dimensions")
        if crop_low.dtype != int or crop_high.dtype != int:
            raise ValueError("The crop values should be int values, instead they are defined as: {} and {}"
                             .format(crop_low.dtype, crop_high.dtype))

    def validate_input_observation_space(self, input_observation_space: ObservationSpace):
        if np.any(self.crop_high > input_observation_space.shape) or \
                np.any(self.crop_low > input_observation_space.shape):
            raise ValueError("The cropping values are outside of the observation space")
        if not input_observation_space.is_point_in_space_shape(self.crop_low) or \
                not input_observation_space.is_point_in_space_shape(self.crop_high - 1):
            raise ValueError("The cropping indices are outside of the observation space")

    def filter(self, observation: ObservationType) -> ObservationType:
        indices = [slice(i, j) for i, j in zip(self.crop_low, self.crop_high)]
        observation = observation[indices]
        return observation

    def get_filtered_observation_space(self, input_observation_space: ObservationSpace) -> ObservationSpace:
        input_observation_space.shape = self.crop_high - self.crop_low
        return input_observation_space
