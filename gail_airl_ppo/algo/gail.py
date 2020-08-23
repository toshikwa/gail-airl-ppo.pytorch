import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from .ppo import PPO
from gail_airl_ppo.network import GAILDiscrim


class GAIL(PPO):

    def __init__(self, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.995, rollout_length=50000, mix_buffer=1,
                 batch_size=64, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc=(100, 100), epoch_ppo=50, epoch_disc=10,
                 clip_eps=0.2, lambd=0.97, coef_ent=0.0, max_grad_norm=10.0):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = GAILDiscrim(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_disc,
            hidden_activation=nn.Tanh()
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def update(self, writer):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            states, actions = self.buffer.sample(self.batch_size)[:2]
            # Samples from expert's demonstrations.
            states_exp, actions_exp = \
                self.buffer_exp.sample(self.batch_size)[:2]
            # Update discriminator.
            self.update_disc(states, actions, states_exp, actions_exp, writer)

        # We don't use reward signals here,
        states, actions, _, dones, log_pis, next_states = self.buffer.get()

        # Calculate rewards.
        rewards = self.disc.calculate_reward(states, actions)

        # Update PPO using estimated rewards.
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_disc(self, states, actions, states_exp, actions_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, actions)
        logits_exp = self.disc(states_exp, actions_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)
