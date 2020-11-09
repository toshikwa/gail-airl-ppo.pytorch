import os
import numpy as np
import torch


class SerializedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)

        self.traj_states = []
        self.traj_actions = []
        self.traj_rewards = []

        self.n_traj = 0
        traj_states = torch.Tensor([]).to(self.device)
        traj_actions = torch.Tensor([]).to(self.device)
        traj_rewards = 0
        for i, done in enumerate(self.dones):
            traj_states = torch.cat((traj_states, self.states[i]), dim=0)
            traj_actions = torch.cat((traj_actions, self.actions[i]), dim=0)
            traj_rewards += self.rewards[i]
            if done == 1:
                self.traj_states.append(traj_states)
                self.traj_actions.append(traj_actions)
                self.traj_rewards.append(traj_rewards)
                traj_states = torch.Tensor([]).to(self.device)
                traj_actions = torch.Tensor([]).to(self.device)
                traj_rewards = 0
                self.n_traj += 1

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )

    def get(self):
        return (
            self.states,
            self.actions,
            self.rewards,
            self.dones,
            self.next_states
        )

    def sample_traj(self, batch_size):
        idxes = np.random.randint(low=0, high=self.n_traj, size=batch_size)
        return (
            self.traj_states[idxes],
            self.traj_actions[idxes],
            self.traj_rewards[idxes]
        )


class Buffer(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)


class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.device = device
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample_traj(self, batch_size):
        assert self._p % self.buffer_size == 0

        n_traj = 0
        all_traj_states = []
        all_traj_actions = []
        all_traj_rewards = []
        traj_states = torch.Tensor([]).to(self.device)
        traj_actions = torch.Tensor([]).to(self.device)
        traj_rewards = 0
        for i, done in enumerate(self.dones):
            traj_states = torch.cat((traj_states, self.states[i].unsqueeze(0)), dim=0)
            traj_actions = torch.cat((traj_actions, self.actions[i].unsqueeze(0)), dim=0)
            traj_rewards += self.rewards[i]
            if done == 1:
                all_traj_states.append(traj_states)
                all_traj_actions.append(traj_actions)
                all_traj_rewards.append(traj_rewards)
                traj_states = torch.Tensor([]).to(self.device)
                traj_actions = torch.Tensor([]).to(self.device)
                traj_rewards = 0
                n_traj += 1

        idxes = np.random.randint(low=0, high=n_traj, size=batch_size)
        return (
            np.array(all_traj_states)[idxes],
            np.array(all_traj_actions)[idxes],
            np.array(all_traj_rewards)[idxes]
        )
