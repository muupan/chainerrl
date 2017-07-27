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
    """Hook function that will be called in training.

    This class is for clarifying the interface required for Hook functions.
    You don't need to inherit this class to define your own hooks. Any callable
    that accepts (env, agent, step) as arguments can be used as a hook.
    """

    @abstractmethod
    def __call__(self, env, agent, step):
        """Call the hook.

        Args:
            env: Environment.
            agent: Agent.
            step: Current timestep.
        """
        raise NotImplementedError


class LinearInterpolationHook(StepHook):
    """Hook that will set a linearly interpolated value.

    You can use this hook to decay the learning rate by using a setter function
    as follows:

    .. code-block:: python

        def lr_setter(env, agent, value):
            agent.optimizer.lr = value

        hook = LinearInterpolationHook(10 ** 6, 1e-3, 0, lr_setter)


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


class PiecewiseLinearInterpolationHook(StepHook):
    """Hook that will set a piecewise-linearly interpolated value.

    You can use this hook to decay the exploration parameter by using a setter
    function as follows:

    .. code-block:: python

        def epsilon_setter(env, agent, value):
            agent.explorer.epsilon = value

        hook = PiecewiseLinearInterpolationHook(
            1.0, [(10 ** 6, 0.1), (10 ** 7, 0.01)], epsilon_setter)


    Args:
        total_steps (int): Number of total steps.
        start_value (float): Start value.
        schedule (list): List of tuples (step, value).
        setter (callable): (env, agent, value) -> None
    """

    def __init__(self, start_value, schedule, setter):
        self.start_value = start_value
        self.schedule = schedule
        self.setter = setter

    def __call__(self, env, agent, step):
        assert step > 0
        if not self.schedule:
            self.setter(env, agent, self.start_value)
        last_piece_steps = 1
        last_piece_value = self.start_value
        for scheduled_steps, scheduled_value in self.schedule:
            assert last_piece_steps <= step
            assert last_piece_steps <= scheduled_steps
            if step <= scheduled_steps:
                value = np.interp(step,
                                  [last_piece_steps, scheduled_steps],
                                  [last_piece_value, scheduled_value])
                self.setter(env, agent, value)
                return
            else:
                last_piece_steps = scheduled_steps
                last_piece_value = scheduled_value
        return last_piece_value
