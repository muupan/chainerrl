import gym
gym.undo_logger_setup()
from gym import spaces

import numpy as np


def flatten_space(obs_space):
    """Flatten a space into a 1D Box space."""
    if isinstance(obs_space, spaces.Box):
        return spaces.Box(low=obs_space.low.ravel(),
                          high=obs_space.high.ravel())
    elif isinstance(obs_space, spaces.Discrete):
        return spaces.Box(low=np.zeros(obs_space.n),
                          high=np.ones(obs_space.n))
    elif isinstance(obs_space, spaces.Tuple):
        flattened_children = [flatten_space(child)
                              for child in obs_space.spaces]
        return spaces.Box(
            low=np.concatenate([child.low for child in flattened_children]),
            high=np.concatenate([child.high for child in flattened_children]))
    else:
        assert False, '{} cannot be flattened'.format(obs_space)


def flatten_value_in_space(obs_space, value):
    """Flatten a value into a 1D Box space."""
    if isinstance(obs_space, spaces.Box):
        return value.ravel()
    elif isinstance(obs_space, spaces.Discrete):
        onehot = np.zeros(obs_space.n, dtype=np.float32)
        onehot[value] = 1
        return onehot
    elif isinstance(obs_space, spaces.Tuple):
        return np.concatenate(
            [flatten_value_in_space(s, v)
             for s, v in zip(obs_space.spaces, value)])
    else:
        assert False, '{} cannot be flattened'.format(obs_space)


class FlattenObservation(gym.ObservationWrapper):
    """Flatten observation space into a 1D Box space.

    The behaviour of this wrapper depends on the base env's observation space:
        Box: its shape is flattened.
        Discrete: it's replaced with a Box that consists of one-hot
        Tuple: it's replaced with a Box that concatenates flattened values in
            that tuple.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = flatten_space(env.observation_space)

    def _observation(self, observation):
        return flatten_value_in_space(self.env.observation_space, observation)
