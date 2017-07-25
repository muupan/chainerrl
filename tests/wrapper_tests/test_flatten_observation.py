from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest

import gym
from gym import spaces

from chainer import testing

import chainerrl


class TestFlattenObservation(unittest.TestCase):

    def setUp(self):
        pass

    def test_frozen_lake(self):
        # Discrete(4) -> Box((4,))
        env = gym.make('FrozenLake-v0')
        self.assertIsInstance(env.observation_space, spaces.Discrete)
        self.assertEqual(env.observation_space.n, 16)

        env = chainerrl.wrappers.FlattenObservation(env)
        self.assertIsInstance(env.observation_space, spaces.Box)
        self.assertEqual(env.observation_space.low.shape, (16,))

        obs = env.reset()
        self.assertTrue(env.observation_space.contains(obs))

    def test_offswitch_cartpole(self):
        # Tuple(Discrete(2), Box((4,))) -> Box((6,))
        env = gym.make('OffSwitchCartpole-v0')
        self.assertIsInstance(env.observation_space, spaces.Tuple)
        space0, space1 = env.observation_space.spaces
        self.assertIsInstance(space0, spaces.Discrete)
        self.assertEqual(space0.n, 2)
        self.assertIsInstance(space1, spaces.Box)
        self.assertEqual(space1.low.shape, (4,))

        env = chainerrl.wrappers.FlattenObservation(env)
        self.assertIsInstance(env.observation_space, spaces.Box)
        self.assertEqual(env.observation_space.low.shape, (6,))
        obs = env.reset()
        self.assertTrue(env.observation_space.contains(obs))


testing.run_module(__name__, __file__)
