import logging

import chainer
from chainer import cuda
import chainer.functions as F
import copy

import numpy as np


import chainerrl
from chainerrl import agent
from chainerrl.misc.batch_states import batch_states

from chainerrl.misc.weighted_std import weighted_std

from chainerrl.recurrent import state_reset


def _F_clip(x, x_min, x_max):
    """Elementwise clipping

    Note: chainer.functions.clip supports clipping to constant intervals
    """
    return F.minimum(F.maximum(x, x_min), x_max)


def compute_mean_and_std_of_advantage(transitions):
    advs = np.asarray([float(b['adv']) for b in transitions])
    weights = np.asarray([float(b['weight']) for b in transitions])
    mean = np.average(advs, weights=weights)
    std = weighted_std(advs, weights=weights) + 1e-8
    assert std > 0
    return mean, std


def normalize_advantage(transitions, mean, std):
    assert std > 0
    for b in transitions:
        b['adv'] = (b['adv'] - mean) / std


def clip_advantage(transitions, max_abs_advantage):
    assert max_abs_advantage > 0
    for b in transitions:
        b['adv'] = b['adv'].clip(min=-max_abs_advantage, max=max_abs_advantage)


def merge_model_params(model_from, model_to):
    params_from = sorted(model_from.namedparams(), key=lambda x: x[0])
    params_to = sorted(model_to.namedparams(), key=lambda x: x[0])
    for param_from, param_to in zip(params_from, params_to):
        assert param_from[0] == param_to[0]
        param_to[1].data[:] = 0.5 * param_to[1].data + 0.5 * param_from[1].data


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
                 normalize_advantage_episodewise=False,
                 max_abs_advantage=None,
                 act_deterministically=False,
                 average_v_decay=0.999, average_loss_decay=0.99,
                 recurrent=False,
                 max_kl=None,
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
        self.normalize_advantage_episodewise = normalize_advantage_episodewise
        self.max_abs_advantage = max_abs_advantage
        self.recurrent = recurrent
        self.max_kl = max_kl
        self.logger = logger

        self.xp = self.model.xp
        self.last_state = None
        self.act_deterministically = act_deterministically

        self.memory = []
        self.episodic_memory = []
        self.last_episode = []
        self.n_episodes_in_memory = 0

        self.recorder = chainerrl.statistics_recorder.StatisticsRecorder()
        self.recorder.register('value', maxlen=1000)
        self.recorder.register('vf_loss', maxlen=1000)
        self.recorder.register('policy_loss', maxlen=1000)
        self.recorder.register('policy_entropy', maxlen=10000)
        self.recorder.register('policy_kl', maxlen=10000)
        self.recorder.register('prob_ratio', maxlen=10000)
        self.recorder.register('value_change', maxlen=10000)
        self.recorder.register('explained_variance', maxlen=1000)
        self.recorder.register('raw_advantage', maxlen=10000)
        self.recorder.register('normalized_advantage', maxlen=10000)

    def _act(self, state, train, deterministic):
        xp = self.xp
        with chainer.using_config('train', train), chainer.no_backprop_mode():
            b_state = batch_states([state], xp, self.phi)
            action_distrib, v = self.model(b_state)
            if deterministic:
                if hasattr(action_distrib, 'mean'):
                    action = action_distrib.mean
                else:
                    action = action_distrib.most_probable
            else:
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
            assert len(self.memory) == sum(len(ep)
                                           for ep in self.episodic_memory)
            self.update()
            self.memory = []
            self.episodic_memory = []
            self.n_episodes_in_memory = 0

    def _flush_last_episode(self):
        if self.last_episode:
            self._compute_teacher()
            self.memory.extend(self.last_episode)
            self.episodic_memory.append(self.last_episode)
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
                 advs, vs_teacher, weights):
        prob_ratio = F.exp(log_probs - target_log_probs)
        ent = distribs.entropy

        def weighted_mean(xs):
            # Divide by N, not the sum of weights
            if xs.shape != weights.shape:
                w = F.broadcast_to(weights[..., None], xs.shape)
            else:
                w = weights
            return F.mean(xs * w)

        prob_ratio = F.expand_dims(prob_ratio, axis=-1)
        loss_policy = - weighted_mean(F.minimum(
            prob_ratio * advs,
            F.clip(prob_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs))
        self.recorder.record('prob_ratio', prob_ratio.data)

        if self.clip_eps_vf is None:
            loss_value_func = weighted_mean(
                F.squared_error(vs_pred, vs_teacher))
        else:
            loss_value_func = weighted_mean(F.maximum(
                F.square(vs_pred - vs_teacher),
                F.square(_F_clip(vs_pred,
                                 vs_pred_old - self.clip_eps_vf,
                                 vs_pred_old + self.clip_eps_vf)
                         - vs_teacher))
            )
        self.recorder.record('value_change', vs_pred.data - vs_pred_old)

        loss_entropy = -weighted_mean(ent)

        # Update stats
        self.recorder.record('policy_loss', float(loss_policy.data))
        self.recorder.record('vf_loss', float(loss_value_func.data))
        self.recorder.record('policy_entropy', chainer.cuda.to_cpu(ent.data))

        return (
            loss_policy
            + self.value_func_coeff * loss_value_func
            + self.entropy_coeff * loss_entropy
        )

    def update_from_episodes(self, episodes, target_model):
        xp = self.xp
        with state_reset(self.model), state_reset(target_model):
            loss = 0
            tmp = list(reversed(sorted(
                enumerate(episodes), key=lambda x: len(x[1]))))
            # Desc by lengths of episodes
            sorted_episodes = [elem[1] for elem in tmp]
            indices = [elem[0] for elem in tmp]  # argsort
            max_epi_len = len(sorted_episodes[0])
            for i in range(max_epi_len):
                batch = []
                for ep, index in zip(sorted_episodes, indices):
                    if len(ep) <= i:
                        break
                    batch.append(ep[i])
                # Now batch contains transitions at timestep i
                states = batch_states([b['state']
                                       for b in batch], xp, self.phi)
                actions = xp.array([b['action'] for b in batch])
                weights = xp.array([b['weight']
                                    for b in batch], dtype=np.float32)
                distribs, vs_pred = self.model(states)
                with chainer.no_backprop_mode():
                    target_distribs, _ = target_model(states)
                loss += self._lossfun(
                    distribs, vs_pred, distribs.log_prob(actions),
                    target_distribs=target_distribs,
                    vs_pred_old=xp.array(
                        [b['v_pred'] for b in batch], dtype=xp.float32),
                    target_log_probs=target_distribs.log_prob(actions),
                    advs=xp.array([b['adv'] for b in batch], dtype=xp.float32),
                    vs_teacher=xp.array(
                        [b['v_teacher'] for b in batch], dtype=xp.float32),
                    weights=weights,
                )
            loss /= max_epi_len

            self.model.cleargrads()
            loss.backward()
            self.optimizer.update()

    def update(self):
        self.logger.debug('update memory: %s n_episodes: %s',
                          len(self.memory),
                          self.n_episodes_in_memory)
        assert self.memory
        xp = self.xp
        target_model = copy.deepcopy(self.model)
        # Compute explained variance
        if len(self.memory) > 0:
            adv_var = np.var([float(b['adv']) for b in self.memory])
            ret_var = np.var([float(b['v_teacher']) for b in self.memory])
            self.recorder.record('explained_variance', 1 - adv_var / ret_var)

        raw_advs = np.asarray([float(b['adv']) for b in self.memory])
        self.recorder.record('raw_advantage', raw_advs)
        self.logger.debug(
            'raw_advantage mean: %s std: %s', raw_advs.mean(), raw_advs.std())

        if self.normalize_advantage:
            if self.normalize_advantage_episodewise:
                # Normalize each episode separately
                for episode in self.episodic_memory:
                    mean, std = compute_mean_and_std_of_advantage(episode)
                    self.logger.debug(
                            ('advantage normlization (episode-wise)'
                             ' len: %s mean: %s std: %s'),  # NOQA
                        len(episode), mean, std)
                    normalize_advantage(episode, mean, std)
            else:
                mean, std = compute_mean_and_std_of_advantage(self.memory)
                self.logger.debug(
                    'advantage normlization (global) mean: %s std: %s',
                    mean, std)
                normalize_advantage(self.memory, mean, std)

            if self.max_abs_advantage is not None:
                clip_advantage(self.memory, self.max_abs_advantage)

            new_advs = np.asarray([float(b['adv']) for b in self.memory])
            self.recorder.record('normalized_advantage', new_advs)
            self.logger.debug(
                'normalized_advantage mean: %s std: %s',
                new_advs.mean(), new_advs.std())

        if self.recurrent:
            batch_size = min(self.n_episodes_in_memory, self.minibatch_size)
            dataset_iter = chainer.iterators.SerialIterator(
                self.episodic_memory, batch_size)
            dataset_iter.reset()
            it = 0
            while dataset_iter.epoch < self.epochs:
                self.logger.debug('update recurrent iter: %s epoch: %s',
                                  it, dataset_iter.epoch)
                batch_episodes = dataset_iter.__next__()
                self.update_from_episodes(batch_episodes, target_model)
                it += 1
        else:
            dataset_iter = chainer.iterators.SerialIterator(
                self.memory, self.minibatch_size)
            dataset_iter.reset()
            last_batch = None
            while dataset_iter.epoch < self.epochs:
                batch = dataset_iter.__next__()
                states = batch_states([b['state']
                                       for b in batch], xp, self.phi)
                actions = xp.array([b['action'] for b in batch])
                weights = xp.array([b['weight']
                                    for b in batch], dtype=np.float32)
                distribs, vs_pred = self.model(states)
                with chainer.no_backprop_mode():
                    target_distribs, _ = target_model(states)
                # Compute KL div.
                kl = target_distribs.kl(distribs)
                self.recorder.record('policy_kl', kl.data)
                if self.max_kl is None and kl.data.max() > 1.0:
                    self.logger.warning('max_kl > 1.0')
                    import time
                    log_file = 'max_kl_warning_' + str(time.time()) + '.log'
                    with open(log_file, 'w') as f:
                        print('max_kl:', kl.data.max(), file=f)
                        print('current_batch:', batch, file=f)
                        print('last_batch:', last_batch, file=f)
                        print('stats:', self.get_statistics(), file=f)
                        print('memory:', self.memory, file=f)
                    self.logger.warning('log: %s', log_file)
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
                    weights=weights,
                )
                last_batch = batch
        if self.max_kl is not None:
            all_states = batch_states(
                [b['state'] for b in self.memory], xp, self.phi)
            all_weights = xp.array([b['weight']
                                    for b in batch], dtype=np.float32)
            self.line_search(target_model, all_states, all_weights)

    def line_search(self, old_model, states, weights):
        assert not self.recurrent
        with chainer.no_backprop_mode():
            old_distribs, _ = old_model(states)
            while True:
                distribs, _ = self.model(states)
                kl = old_distribs.kl(distribs)
                # KL div. should be weighted so that samples with zero weight
                # are ignored when computing max KL div.
                cur_max_kl = (kl.data * weights).max()
                self.logger.info('line search kl: %s', cur_max_kl)
                if cur_max_kl < self.max_kl:
                    break
                merge_model_params(model_from=old_model,
                                   model_to=self.model)

    def act_and_train(self, state, reward, weight=1.0):
        action, v = self._act(state, train=False, deterministic=False)

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
                'nonterminal': 1.0,
                'weight': weight,
            })
        self.last_state = state
        self.last_action = action
        self.last_v = v

        self._train()
        return action

    def act(self, state):
        action, v = self._act(state, train=False,
                              deterministic=self.act_deterministically)

        self.recorder.record('value', float(v))
        return action

    def stop_episode_and_train(self, state, reward, done=False, weight=1.0):
        _, v = self._act(state, train=False, deterministic=False)

        assert self.last_state is not None
        self.last_episode.append({
            'state': self.last_state,
            'action': self.last_action,
            'reward': reward,
            'v_pred': self.last_v,
            'next_state': state,
            'next_v_pred': v,
            'nonterminal': 0.0 if done else 1.0,
            'weight': weight,
        })

        self.last_state = None
        del self.last_action
        del self.last_v

        self._flush_last_episode()
        self._train()
        self.stop_episode()

    def stop_episode(self):
        if self.recurrent:
            self.model.reset_state()

    def get_statistics(self):
        return self.recorder.get_statistics()


class ParallelPPO(PPO):

    def batch_act_and_train(self, states, rewards, dones, interrupts, weights):
        n_actors = len(states)
        assert len(rewards) == n_actors
        assert len(dones) == n_actors
        assert len(interrupts) == n_actors
        assert not self.recurrent, 'not yet implemented'
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
        self.recorder.record('value', v)
        # Update stats
        # self.average_v += (
        #     (1 - self.average_v_decay) *
        #     (float(v.mean()) - self.average_v))
        # Put experiences into the buffer
        for i in range(len(states)):
            if self.last_state[i] is not None:
                assert rewards[i] is not None
                self.last_episode[i].append({
                    'state': self.last_state[i],
                    'action': self.last_action[i],
                    'reward': rewards[i],
                    'v_pred': self.last_v[i],
                    'next_state': states[i],
                    'next_v_pred': v[i],
                    'nonterminal': 0. if dones[i] else 1.,
                    'uninterrupted': 0. if interrupts[i] else 1.,
                    'weight': weights[i],
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
        assert len(self.memory) == sum(len(ep) for ep in self.episodic_memory)
        self.update()
        self.memory = []
        self.episodic_memory = []
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
            self.episodic_memory.append(seq)
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
                assert reward is not None
                td_err = (
                    reward
                    + (self.gamma * transition['nonterminal']
                       * transition['next_v_pred'])
                    - transition['v_pred']
                )
                # Since seq may contain multiple episodes, you have to
                # truncate GAE when interrupted or terminaal
                adv = (td_err
                       + (transition['uninterrupted']
                           * transition['nonterminal']
                           * self.gamma
                           * self.lambd
                           * adv))
                transition['adv'] = adv
                transition['v_teacher'] = adv + transition['v_pred']
