from typing import List

import numpy as np


class Schedule(object):
    def __init__(self, initial_value: float):
        self.initial_value = initial_value
        self.current_value = initial_value

    def step(self):
        raise NotImplementedError("")


class ConstantSchedule(Schedule):
    def __init__(self, initial_value: float):
        super().__init__(initial_value)

    def step(self):
        pass


class LinearSchedule(Schedule):
    """
    A simple linear schedule which decreases or increases over time from an initial to a final value
    """
    def __init__(self, initial_value: float, final_value: float, decay_steps: int):
        """
        :param initial_value: the initial value
        :param final_value: the final value
        :param decay_steps: the number of steps that are required to decay the initial value to the final value
        """
        super().__init__(initial_value)
        self.final_value = final_value
        self.decay_steps = decay_steps
        self.decay_delta = (initial_value - final_value) / float(decay_steps)

    def step(self):
        self.current_value -= self.decay_delta
        # decreasing schedule
        if self.final_value < self.initial_value:
            self.current_value = np.clip(self.current_value, self.final_value, self.initial_value)
        # increasing schedule
        if self.final_value > self.initial_value:
            self.current_value = np.clip(self.current_value, self.initial_value, self.final_value)


class PieceWiseLinearSchedule(Schedule):
    """
    A schedule which is linear in several ranges
    """
    def __init__(self, schedules: List[LinearSchedule]):
        """
        :param schedules: a list of schedules to apply serially
        """
        super().__init__(schedules[0].initial_value)
        self.schedules = schedules
        self.current_schedule = schedules[0]
        self.current_schedule_idx = 0

    def step(self):
        if self.current_schedule_idx < len(self.schedules) - 1 and \
                        self.current_schedule.final_value == self.current_schedule.current_value:
            self.current_schedule_idx += 1
            self.current_schedule = self.schedules[self.current_schedule_idx]
        self.current_schedule.step()
        self.current_value = self.current_schedule.current_value


class ExponentialSchedule(Schedule):
    """
    A simple exponential schedule which decreases or increases over time from an initial to a final value
    """
    def __init__(self, initial_value: float, final_value: float, decay_coefficient: int):
        """
        :param initial_value: the initial value
        :param final_value: the final value
        :param decay_coefficient: the exponential decay coefficient
        """
        super().__init__(initial_value)
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_coefficient = decay_coefficient
        self.current_step = 0

    def step(self):
        self.current_value = self.initial_value * np.exp(-self.decay_coefficient * self.current_step)

        # decreasing schedule
        if self.final_value < self.initial_value:
            self.current_value = np.clip(self.current_value, self.final_value, self.initial_value)
        # increasing schedule
        if self.final_value > self.initial_value:
            self.current_value = np.clip(self.current_value, self.initial_value, self.final_value)

        self.current_step += 1