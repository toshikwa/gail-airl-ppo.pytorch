import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .base import Algorithm
from gail_ppo.buffer import RolloutBuffer
from gail_ppo.network import StateIndependentPolicy, StateFunction


def calculate_gae(values, rewards, dones, gamma=0.995, lambd=0.997):
    # Calculate TD errors.
    deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    # Calculate lambda-Return.
    targets = gaes + values[:-1]

    return targets, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class PPO(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, gamma=0.995,
                 batch_size=64, lr_actor=3e-4, lr_critic=3e-4,
                 rollout_length=2048, epoch_ppo=10, clip_eps=0.2,
                 lambd=0.97, coef_ent=0.0, max_grad_norm=10.0):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        # Rollout buffer.
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device
        )

        # Actor.
        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=(64, 64),
            hidden_activation=nn.Tanh()
        ).to(self.device)

        # Critic.
        self.critic = StateFunction(
            state_shape=state_shape,
            hidden_units=(64, 64),
            hidden_activation=nn.Tanh()
        ).to(self.device)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.batch_size = batch_size
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def explore(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor(state.unsqueeze_(0))
        return action.cpu().numpy()[0]

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, step):
        t += 1

        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)

        if t == env._max_episode_steps:
            mask = False
        else:
            mask = done

        self.buffer.append(state, action, reward, mask, log_pi)

        if step % self.rollout_length == 0:
            self.buffer.append_last_state(next_state)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self, writer):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis = self.buffer.get()
        self.update_ppo(states, actions, rewards, dones, log_pis, writer)

    def update_ppo(self, states, actions, rewards, dones, log_pis, writer):
        with torch.no_grad():
            values = self.critic(states)

        targets, gaes = calculate_gae(
            values, rewards, dones, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            indices = np.arange(self.rollout_length)
            np.random.shuffle(indices)

            for start in range(0, self.rollout_length, self.batch_size):
                idxes = indices[start:start+self.batch_size]
                self.update_critic(states[idxes], targets[idxes], writer)
                self.update_actor(
                    states[idxes], actions[idxes], log_pis[idxes],
                    gaes[idxes], writer
                )

    def update_critic(self, states, targets, writer):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

    def update_actor(self, states, actions, log_pis_old, gaes, writer):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

    def save_models(self, save_dir):
        pass
