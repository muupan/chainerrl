from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

from abc import ABCMeta
from abc import abstractmethod

from future.utils import with_metaclass
import numpy as np


class StepHook(with_metaclass(ABCMeta, object)):
    """Hook function that will be called in training."""

    @abstractmethod
    def __call__(self, env, agent, step):
        raise NotImplementedError


class LinearInterpolationHook(StepHook):
    """Hook that will set a linearly interpolated value.

    Args:
        total_steps (int): Number of total steps.
        start_value (float): Start value.
        stop_value (float): Stop value.
        setter (callable): (env, agent, value) -> None
    """

    def __init__(self, total_steps, start_value, stop_value, setter):
        self.total_steps = total_steps
        self.start_value = start_value
        self.stop_value = stop_value
        self.setter = setter

    def __call__(self, env, agent, step):
        value = np.interp(step,
                          [1, self.total_steps],
                          [self.start_value, self.stop_value])
        self.setter(env, agent, value)