import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .base import OfflineAlgorithm
from gail_ppo_bcq.network import Perturb, TwinnedStateActionFunction, CVAE
from gail_ppo_bcq.utils import soft_update, disable_gradient


class BCQ(OfflineAlgorithm):

    def __init__(self, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.99, batch_size=100, lr_pert=1e-3, lr_critic=1e-3,
                 lr_cvae=1e-3, units_pert=(400, 300), units_critic=(400, 300),
                 units_cvae=(750, 750), coef_kl=0.5, lambd=0.75, std=0.05,
                 tau=5e-3, n_train=10, n_test=100):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Perturbation model.
        self.pert = Perturb(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_pert,
            hidden_activation=nn.ReLU(inplace=True),
            std=std
        ).to(device)
        self.pert_target = Perturb(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_pert,
            hidden_activation=nn.ReLU(inplace=True),
            std=std
        ).to(device).eval()

        # Critic.
        self.critic = TwinnedStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_critic,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        self.critic_target = TwinnedStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_critic,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device).eval()

        # CVAE．
        self.cvae = CVAE(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_cvae,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)

        soft_update(self.pert_target, self.pert, 1.0)
        soft_update(self.critic_target, self.critic, 1.0)
        disable_gradient(self.pert_target)
        disable_gradient(self.critic_target)

        self.optim_pert = Adam(self.pert.parameters(), lr=lr_pert)
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

    def explore(self, state):
        NotImplementedError

    def exploit(self, state):
        state = torch.tensor(
            np.tile(state, (self.n_test, 1)),
            dtype=torch.float, device=self.device
        )
        with torch.no_grad():
            # Generate action candidates.
            action = self.cvae.generate(state)
            # Perturb actions.
            action = self.pert(state, action)
            # Select action with maximal Q.
            action_index = self.critic.q1(state, action).argmax()
        return action[action_index].cpu().numpy()

    def update(self, writer):
        self.learning_steps += 1
        states, actions, rewards, dones, next_states = \
            self.buffer_exp.sample(self.batch_size)

        self.update_cvae(states, actions, writer)
        self.update_critic(
            states, actions, rewards, dones, next_states, writer)
        self.update_pert(states, writer)
        self.update_target()

    def update_cvae(self, states, actions, writer):
        reconsts, means, log_vars = self.cvae(states, actions)

        loss_reconst = (reconsts - actions).pow_(2).sum(dim=1).mean()
        loss_kl = -0.5 * (
            log_vars - means.pow(2) - log_vars.exp()
        ).sum(dim=1).mean()
        loss_cvae = loss_reconst + self.coef_kl * loss_kl

        self.optim_cvae.zero_grad()
        loss_cvae.backward(retain_graph=False)
        self.optim_cvae.step()

        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/reconst', loss_reconst.item(), self.learning_steps)
            writer.add_scalar(
                'loss/kl', loss_kl.item(), self.learning_steps)

    def update_critic(self, states, actions, rewards, dones, next_states,
                      writer):
        curr_qs1, curr_qs2 = self.critic(states, actions)

        with torch.no_grad():
            # Expand tensor to calculate N action candidates.
            next_states = next_states.repeat_interleave(self.n_train, 0)
            # Generate action candidates.
            next_actions = self.cvae.generate(next_states)
            # Perturb actions.
            next_actions = self.pert_target(next_states, next_actions)
            # Calculate next Q.
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

        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/critic1', loss_critic1.item(), self.learning_steps)
            writer.add_scalar(
                'loss/critic2', loss_critic2.item(), self.learning_steps)

    def update_pert(self, states, writer):
        actions = self.cvae.generate(states)
        actions = self.pert(states, actions)
        loss_pert = -self.critic.q1(states, actions).mean()

        self.optim_pert.zero_grad()
        loss_pert.backward(retain_graph=False)
        self.optim_pert.step()

        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/perturb', loss_pert.item(), self.learning_steps)

    def update_target(self):
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.pert_target, self.pert, self.tau)

    def save_models(self, save_dir):
        pass
