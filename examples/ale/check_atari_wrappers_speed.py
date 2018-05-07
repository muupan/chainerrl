from timeit import default_timer as timer

import numpy as np

from chainerrl.envs import ale

import atari_wrappers
import dqn_phi


old_phi = dqn_phi.dqn_phi


def new_phi(x):
    return np.asarray(x, dtype=np.float32) / 255


def check_speed(env, phi):
    start = timer()
    obs = env.reset()
    done = False
    for _ in range(10000):
        x = phi(obs)
        assert x.shape == (4, 84, 84)
        obs, r, done, info = env.step(0)
        if done:
            print('.', end='', flush=True)
            obs = env.reset()
    print('')
    print(timer() - start)


# rom_name = 'pong'
# env_name = 'PongNoFrameskip-v4'
rom_name = 'breakout'
env_name = 'BreakoutNoFrameskip-v4'

print('rom name:', rom_name)
print('env name:', env_name)


print('old env')
for _ in range(3):
    old_env = ale.ALE(rom_name, seed=0)
    check_speed(old_env, old_phi)

print('new env')
for _ in range(3):
    new_env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari(env_name))
    new_env.seed(0)
    check_speed(new_env, new_phi)

print('old env (test)')
for _ in range(3):
    old_env = ale.ALE(rom_name, treat_life_lost_as_terminal=False, seed=0)
    check_speed(old_env, old_phi)

print('new env (test)')
for _ in range(3):
    new_env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari(env_name),
        episode_life=False,
        clip_rewards=False,
    )
    new_env.seed(0)
    check_speed(new_env, new_phi)
