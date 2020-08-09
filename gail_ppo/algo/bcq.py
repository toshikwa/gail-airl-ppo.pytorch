import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .base import Algorithm
from gail_ppo.network import Noise, TwinnedStateActionFunction, CVAE
from gail_ppo.utils import soft_update, disable_gradient


class BCQ(Algorithm):

    def __init__(self, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.99, batch_size=100, lr_noise=1e-3, lr_critic=1e-3,
                 lr_cvae=1e-3, coef_kl=0.5, lambd=0.75, std=0.05, tau=5e-3,
                 n_train=10, n_test=100):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Noise.
        self.noise = Noise(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=(400, 300),
            hidden_activation=nn.ReLU(inplace=True),
            std=std
        ).to(device)
        self.noise_target = Noise(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=(400, 300),
            hidden_activation=nn.ReLU(inplace=True),
            std=std
        ).to(device).eval()

        # Critic.
        self.critic = TwinnedStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=(400, 300),
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        self.critic_target = TwinnedStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=(400, 300),
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device).eval()

        # CVAE．
        self.cvae = CVAE(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=(750, 750),
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)

        soft_update(self.noise_target, self.noise, 1.0)
        soft_update(self.critic_target, self.critic, 1.0)
        disable_gradient(self.noise_target)
        disable_gradient(self.critic_target)

        self.optim_noise = Adam(self.noise.parameters(), lr=lr_noise)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)
        self.optim_cvae = Adam(self.cvae.parameters(), lr=lr_cvae)

        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.coef_kl = coef_kl
        self.lambd = lambd
        self.tau = tau
        self.n_train = n_train
        self.n_test = n_test

    def exploit(self, state):
        state = torch.tensor(
            np.tile(state, (self.n_test, 1)),
            dtype=torch.float, device=self.device
        )
        with torch.no_grad():
            action = self.cvae.generate(state)
            action = self.noise(state, action)
            action_index = self.critic.q1(state, action).argmax()
        return action[action_index].cpu().numpy()

    def update(self, writer):
        states, actions, rewards, dones, next_states = \
            self.buffer_exp.sample(self.batch_size)

        self.update_cvae(states, actions)
        self.update_critic(states, actions, rewards, dones, next_states)
        self.update_noise(states)
        self.update_target()

    def update_cvae(self, states, actions):
        reconsts, means, log_vars = self.cvae(states, actions)

        loss_reconst = (reconsts - actions).pow_(2).sum(dim=1).mean()
        loss_kl = -0.5 * (
            1 + log_vars - means.pow(2) - log_vars.exp()
        ).sum(dim=1).mean()
        loss_cvae = loss_reconst + self.coef_kl * loss_kl

        self.optim_cvae.zero_grad()
        loss_cvae.backward(retain_graph=False)
        self.optim_cvae.step()

    def update_critic(self, states, actions, rewards, dones, next_states):
        curr_qs1, curr_qs2 = self.critic(states, actions)

        with torch.no_grad():
            next_states = next_states.repeat_interleave(self.n_train, 0)
            next_actions = self.cvae.generate(next_states)
            next_actions = self.noise_target(next_states, next_actions)
            next_qs1, next_qs2 = self.critic_target(next_states, next_actions)

        # Clipped Double Q.
        next_qs = self.lambd * torch.min(next_qs1, next_qs2)
        next_qs.add_((1.0 - self.lambd) * torch.max(next_qs1, next_qs2))
        # Select max Q from n candidates.
        next_qs = next_qs.view(-1, self.n_train).max(dim=1, keepdims=True)[0]
        # Calculate target Q.
        target_qs = rewards + (1 - dones) * self.gamma * next_qs

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_critic.step()

    def update_noise(self, states):
        actions = self.cvae.generate(states)
        actions = self.noise(states, actions)
        loss_noise = -self.critic.q1(states, actions).mean()

        self.optim_noise.zero_grad()
        loss_noise.backward(retain_graph=False)
        self.optim_noise.step()

    def update_target(self):
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.noise_target, self.noise, self.tau)

    def save_models(self, save_dir):
        pass
