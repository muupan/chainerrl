import logging

import chainer
from chainer import cuda
import chainer.functions as F
import copy

import numpy as np


import chainerrl
from chainerrl import agent
from chainerrl.misc.batch_states import batch_states


def _F_clip(x, x_min, x_max):
    """Elementwise clipping

    Note: chainer.functions.clip supports clipping to constant intervals
    """
    return F.minimum(F.maximum(x, x_min), x_max)


class PPO(agent.AttributeSavingMixin, agent.Agent):
    """Proximal Policy Optimization

    See https://arxiv.org/abs/1707.06347

    Args:
        model (A3CModel): Model to train.  Recurrent models are not supported.
            state s  |->  (pi(s, _), v(s))
        optimizer (chainer.Optimizer): optimizer used to train the model
        gpu (int): GPU device id if not None nor negative
        gamma (float): Discount factor [0, 1]
        lambd (float): Lambda-return factor [0, 1]
        phi (callable): Feature extractor function
        value_func_coeff (float): Weight coefficient for loss of
            value function (0, inf)
        entropy_coeff (float): Weight coefficient for entropoy bonus [0, inf)
        update_interval (int): Model update interval in step
        minibatch_size (int): Minibatch size
        epochs (int): Training epochs in an update
        clip_eps (float): Epsilon for pessimistic clipping of likelihood ratio
            to update policy
        clip_eps_vf (float): Epsilon for pessimistic clipping of value
            to update value function. If it is ``None``, value function is not
            clipped on updates.
        average_v_decay (float): Decay rate of average V, only used for
            recording statistics
        average_loss_decay (float): Decay rate of average loss, only used for
            recording statistics
    """

    saved_attributes = ['model', 'optimizer']

    def __init__(self, model, optimizer,
                 gpu=None,
                 gamma=0.99,
                 lambd=0.95,
                 phi=lambda x: x,
                 value_func_coeff=1.0,
                 entropy_coeff=0.01,
                 update_interval=2048,
                 update_interval_episodes=-1,
                 minibatch_size=64,
                 epochs=10,
                 clip_eps=0.2,
                 clip_eps_vf=0.2,
                 normalize_advantage=True,
                 average_v_decay=0.999, average_loss_decay=0.99,
                 logger=logging.getLogger(__name__),
                 ):
        self.model = model

        if gpu is not None and gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            self.model.to_gpu(device=gpu)

        self.optimizer = optimizer
        self.gamma = gamma
        self.lambd = lambd
        self.phi = phi
        self.value_func_coeff = value_func_coeff
        self.entropy_coeff = entropy_coeff
        self.update_interval = update_interval
        self.update_interval_episodes = update_interval_episodes
        assert ((self.update_interval > 0)
                + (self.update_interval_episodes > 0)) == 1
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.clip_eps = clip_eps
        self.clip_eps_vf = clip_eps_vf
        self.normalize_advantage = normalize_advantage
        self.logger = logger

        # self.average_v = 0
        # self.average_v_decay = average_v_decay
        # self.average_loss_policy = 0
        # self.average_loss_value_func = 0
        # self.average_loss_entropy = 0
        # self.average_loss_decay = average_loss_decay

        self.xp = self.model.xp
        self.target_model = None
        self.last_state = None

        self.memory = []
        self.last_episode = []
        self.n_episodes_in_memory = 0

        self.recorder = chainerrl.statistics_recorder.StatisticsRecorder()
        self.recorder.register('value', maxlen=1000)
        self.recorder.register('vf_loss', maxlen=1000)
        self.recorder.register('policy_loss', maxlen=1000)
        self.recorder.register('policy_entropy', maxlen=1000)
        self.recorder.register('policy_kl', maxlen=1000)
        self.recorder.register('prob_ratio', maxlen=1000)
        self.recorder.register('value_change', maxlen=1000)
        self.recorder.register('explained_variance', maxlen=100)

    def _act(self, state, train):
        xp = self.xp
        with chainer.using_config('train', train), chainer.no_backprop_mode():
            b_state = batch_states([state], xp, self.phi)
            action_distrib, v = self.model(b_state)
            action = action_distrib.sample()
            self.logger.debug('act action: %s distrib: %s v: %s',
                              action.data[0], action_distrib, float(v.data[0]))
            return cuda.to_cpu(action.data)[0], v.data[0]

    def _train(self):
        if self.update_interval > 0 and len(self.memory) + len(self.last_episode) < self.update_interval:
            return
        if self.update_interval_episodes > 0 and self.n_episodes_in_memory < self.update_interval_episodes:
            return
        if len(self.memory) + len(self.last_episode) >= self.update_interval:
            self._flush_last_episode()
            self.update()
            self.memory = []
            self.n_episodes_in_memory = 0

    def _flush_last_episode(self):
        if self.last_episode:
            self._compute_teacher()
            self.memory.extend(self.last_episode)
            self.last_episode = []
            self.n_episodes_in_memory += 1

    def _compute_teacher(self):
        """Estimate state values and advantages of self.last_episode

        TD(lambda) estimation
        """

        adv = 0.0
        for transition in reversed(self.last_episode):
            td_err = (
                transition['reward']
                + (self.gamma * transition['nonterminal']
                   * transition['next_v_pred'])
                - transition['v_pred']
            )
            adv = td_err + self.gamma * self.lambd * adv
            transition['adv'] = adv
            transition['v_teacher'] = adv + transition['v_pred']

    def _lossfun(self,
                 distribs, vs_pred, log_probs,
                 target_distribs, vs_pred_old, target_log_probs,
                 advs, vs_teacher):
        prob_ratio = F.exp(log_probs - target_log_probs)
        ent = distribs.entropy

        prob_ratio = F.expand_dims(prob_ratio, axis=-1)
        loss_policy = - F.mean(F.minimum(
            prob_ratio * (advs + vs_pred_old - vs_pred.data),
            F.clip(prob_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs))
        self.recorder.record('prob_ratio', prob_ratio.data)
        kl = target_distribs.kl(distribs)
        self.recorder.record('policy_kl', kl.data)

        if self.clip_eps_vf is None:
            loss_value_func = F.mean_squared_error(vs_pred, vs_teacher)
        else:
            loss_value_func = F.mean(F.maximum(
                F.square(vs_pred - vs_teacher),
                F.square(_F_clip(vs_pred,
                                 vs_pred_old - self.clip_eps_vf,
                                 vs_pred_old + self.clip_eps_vf)
                         - vs_teacher)
            ))
        self.recorder.record('value_change', vs_pred.data - vs_pred_old)

        loss_entropy = -F.mean(ent)

        # Update stats
        self.recorder.record('policy_loss', float(loss_policy.data))
        self.recorder.record('vf_loss', float(loss_value_func.data))
        self.recorder.record('policy_entropy', -float(loss_entropy.data))

        return (
            loss_policy
            + self.value_func_coeff * loss_value_func
            + self.entropy_coeff * loss_entropy
        )

    def update(self):
        self.logger.debug('update memory: %s', len(self.memory))
        assert self.memory
        xp = self.xp
        target_model = copy.deepcopy(self.model)
        dataset_iter = chainer.iterators.SerialIterator(
            self.memory, self.minibatch_size)

        # Compute explained variance
        if len(self.memory) > 0:
            adv_var = np.var([float(b['adv']) for b in self.memory])
            ret_var = np.var([float(b['v_teacher']) for b in self.memory])
            self.recorder.record('explained_variance', 1 - adv_var / ret_var)

        if self.normalize_advantage:
            advs = np.asarray([float(b['adv']) for b in self.memory])
            mean = np.mean(advs)
            std = np.std(advs)
            self.logger.debug(
                'advantage normlization mean: %s std: %s', mean, std)
            for b in self.memory:
                b['adv'] = (b['adv'] - mean) / std

        dataset_iter.reset()
        while dataset_iter.epoch < self.epochs:
            batch = dataset_iter.__next__()
            states = batch_states([b['state'] for b in batch], xp, self.phi)
            actions = xp.array([b['action'] for b in batch])
            distribs, vs_pred = self.model(states)
            with chainer.no_backprop_mode():
                target_distribs, _ = target_model(states)
            self.optimizer.update(
                self._lossfun,
                distribs, vs_pred, distribs.log_prob(actions),
                target_distribs=target_distribs,
                vs_pred_old=xp.array(
                    [b['v_pred'] for b in batch], dtype=xp.float32),
                target_log_probs=target_distribs.log_prob(actions),
                advs=xp.array([b['adv'] for b in batch], dtype=xp.float32),
                vs_teacher=xp.array(
                    [b['v_teacher'] for b in batch], dtype=xp.float32),
            )

    def act_and_train(self, state, reward):
        action, v = self._act(state, train=False)

        # Update stats
        self.recorder.record('value', float(v))

        if self.last_state is not None:
            self.last_episode.append({
                'state': self.last_state,
                'action': self.last_action,
                'reward': reward,
                'v_pred': self.last_v,
                'next_state': state,
                'next_v_pred': v,
                'nonterminal': 1.0})
        self.last_state = state
        self.last_action = action
        self.last_v = v

        self._train()
        return action

    def act(self, state):
        action, v = self._act(state, train=False)

        self.recorder.record('value', float(v))
        # Update stats
        # self.average_v += (
        #     (1 - self.average_v_decay) *
        #     (float(v) - self.average_v))

        return action

    def stop_episode_and_train(self, state, reward, done=False):
        _, v = self._act(state, train=False)

        assert self.last_state is not None
        self.last_episode.append({
            'state': self.last_state,
            'action': self.last_action,
            'reward': reward,
            'v_pred': self.last_v,
            'next_state': state,
            'next_v_pred': v,
            'nonterminal': 0.0 if done else 1.0})

        self.last_state = None
        del self.last_action
        del self.last_v

        self._flush_last_episode()
        self._train()
        self.stop_episode()

    def stop_episode(self):
        pass

    def get_statistics(self):
        return self.recorder.get_statistics()


class ParallelPPO(PPO):

    def batch_act_and_train(self, states, rewards, dones, interrupts):
        n_actors = len(states)
        assert len(rewards) == n_actors
        assert len(dones) == n_actors
        assert len(interrupts) == n_actors
        # Initialization depending on the number of actors
        if self.last_state is None or len(self.last_state) < n_actors:
            if self.last_state is not None:
                self.logger.debug('Number of actors increased from %s to %s.',
                                  len(self.last_state), n_actors)
            self.last_state = [None] * n_actors
            self.last_action = [None] * n_actors
            self.last_v = [None] * n_actors
            self.last_episode = [[] for _ in range(n_actors)]
        # Select an action for each state
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            b_state = batch_states(states, self.xp, self.phi)
            action_distrib, v = self.model(b_state)
            actions = cuda.to_cpu(action_distrib.sample().data)
            v = cuda.to_cpu(v.data)
        # Update stats
        self.average_v += (
            (1 - self.average_v_decay) *
            (float(v.mean()) - self.average_v))
        # Put experiences into the buffer
        for i in range(len(states)):
            if self.last_state[i] is not None:
                self.last_episode[i].append({
                    'state': self.last_state[i],
                    'action': self.last_action[i],
                    'reward': rewards[i],
                    'v_pred': self.last_v[i],
                    'next_state': states[i],
                    'next_v_pred': v[i],
                    'nonterminal': 0. if dones[i] else 1.,
                    'uninterrupted': 0. if interrupts[i] else 1.,
                })
            if dones[i] or interrupts[i]:
                self.last_state[i] = None
                self.last_action[i] = None
                self.last_v[i] = None
            else:
                self.last_state[i] = states[i]
                self.last_action[i] = actions[i]
                self.last_v[i] = v[i]

        if self.update_interval_episodes > 0:
            self._batch_flush_last_episode(True)
        self._batch_train()
        return actions

    def _batch_train(self):
        n_transitions = sum(len(ep) for ep in self.last_episode)
        if self.update_interval > 0 and n_transitions < self.update_interval:
            return
        if self.update_interval_episodes > 0 and self.n_episodes_in_memory < self.update_interval_episodes:
            return
        if self.update_interval > 0:
            self._batch_flush_last_episode()
        self.update()
        self.memory = []
        self.n_episodes_in_memory = 0

    def _batch_flush_last_episode(self, finished_episodes_only=False):
        self._batch_compute_teacher(finished_episodes_only)
        for i, seq in enumerate(self.last_episode):
            if not seq:
                continue
            if (finished_episodes_only
                    and (seq[-1]['nonterminal']
                         + seq[-1]['uninterrupted']) == 2.):
                continue
            assert isinstance(seq, list)
            assert seq
            old_memory_size = len(self.memory)
            self.memory.extend(seq)
            self.n_episodes_in_memory += 1
            self.last_episode[i] = []
            self.logger.debug('memory extend: %s + %s = %s n_episodes_in_memory: %s',
                              old_memory_size, len(seq), len(self.memory), self.n_episodes_in_memory)

    def _batch_compute_teacher(self, finished_episodes_only=False):
        """Estimate state values and advantages of self.last_episode

        TD(lambda) estimation
        """
        for seq in self.last_episode:
            if not seq:
                continue
            if (finished_episodes_only
                    and (seq[-1]['nonterminal']
                         + seq[-1]['uninterrupted']) == 2.):
                continue
            adv = 0.0
            for transition in reversed(seq):
                reward = transition['reward']
                if reward is None:
                    # this must be an initial state
                    reward = 0.
                td_err = (
                    reward
                    + (self.gamma * transition['nonterminal']
                       * transition['next_v_pred'])
                    - transition['v_pred']
                )
                adv = (td_err
                       + (transition['uninterrupted']
                           * transition['nonterminal']
                           * self.gamma
                           * self.lambd
                           * adv))
                transition['adv'] = adv
                transition['v_teacher'] = adv + transition['v_pred']
