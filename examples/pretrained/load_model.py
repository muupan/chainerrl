
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import argparse
from distutils.util import strtobool
import filelock
import hashlib
import os
import shutil
import tempfile
import time
import sys
from six.moves.urllib import request

from chainer.dataset.download import get_dataset_directory
from chainer.dataset.download import get_dataset_root
from chainer import links as L
from chainer import optimizers
import numpy as np

import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chainerrl import agents
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl import replay_buffer

from chainerrl.wrappers import atari_wrappers
import json
from pdb import set_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='ChopperCommandNoFrameskip-v4',
                        help='OpenAI Atari domain to perform algorithm on.')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    parser.add_argument('--steps', type=int, default=5 * 10 ** 7,
                        help='Total number of timesteps to train the agent.')
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4,
                        help='Minimum replay buffer size before ' +
                        'performing gradient updates.')
    parser.add_argument('--eval-n-steps', type=int, default=125000)
    parser.add_argument('--eval-interval', type=int, default=250000)
    parser.add_argument('--n-best-episodes', type=int, default=30)
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.logging_level)

    # Set a random seed used in ChainerRL.
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2 ** 31 - 1 - args.seed

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    def make_env(test):
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=None),
            episode_life=not test,
            clip_rewards=not test)
        env.seed(int(env_seed))
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = chainerrl.wrappers.RandomizeAction(env, 0.05)
        if args.monitor:
            env = chainerrl.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation' if test else 'training')
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)

    n_actions = env.action_space.n
    q_func = links.Sequence(
        links.NatureDQNHead(),
        L.Linear(512, n_actions),
        DiscreteActionValue)

    # Draw the computational graph and save it in the output directory.
    chainerrl.misc.draw_computational_graph(
        [q_func(np.zeros((4, 84, 84), dtype=np.float32)[None])],
        os.path.join(args.outdir, 'model'))

    # Use the same hyperparameters as the Nature paper
    opt = optimizers.RMSpropGraves(
        lr=2.5e-4, alpha=0.95, momentum=0.0, eps=1e-2)

    opt.setup(q_func)

    rbuf = replay_buffer.ReplayBuffer(10 ** 6)

    explorer = explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0, end_epsilon=0.1,
        decay_steps=10 ** 6,
        random_action_func=lambda: np.random.randint(n_actions))

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    Agent = agents.DQN
    agent = Agent(q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
                  explorer=explorer, replay_start_size=args.replay_start_size,
                  target_update_interval=10 ** 4,
                  clip_delta=True,
                  update_interval=4,
                  batch_accumulator='sum',
                  phi=phi)

    model_path = misc.download_model("DQN", args.env, model_type="best")
    agent.load(model_path)

    eval_stats = experiments.eval_performance(
        env=eval_env,
        agent=agent,
        n_steps=args.eval_n_steps,
        n_episodes=None)
    print('n_episodes: {} mean: {} median: {} stdev {}'.format(
        eval_stats['episodes'],
        eval_stats['mean'],
        eval_stats['median'],
        eval_stats['stdev']))


if __name__ == '__main__':
    main()
