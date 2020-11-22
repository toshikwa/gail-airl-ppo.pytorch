import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn.utils.convert_parameters import parameters_to_vector
import numpy as np
import itertools
import os

from .ppo import PPO
from gail_airl_ppo.network import AIRLDiscrim, AIRLDetachedDiscrim


class SIL(PPO):

    def __init__(self, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.995, rollout_length=10000, mix_buffer=1,
                 batch_size=100, traj_batch_size=20, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc_r=(100, 100), units_disc_v=(100, 100),
                 epoch_ppo=50, epoch_disc=10, clip_eps=0.2, lambd=0.97,
                 coef_ent=0.0, max_grad_norm=10.0, lr_conf=1e-1):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = AIRLDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)
        self.detached_disc = AIRLDetachedDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            device=device,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        )

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.lr_disc = lr_disc
        self.epoch_disc = epoch_disc

        # Confidece net
        self.conf = torch.ones(self.buffer_exp.buffer_size, 1).to(device)

        self.learning_steps_conf = 0
        self.lr_conf = lr_conf
        self.epoch_conf = self.epoch_disc

        self.batch_size = batch_size
        self.traj_batch_size = traj_batch_size

    def sample_exp(self, batch_size):
        # Samples from expert's demonstrations.
        all_states_exp, all_actions_exp, _, all_dones_exp, all_next_states_exp = \
            self.buffer_exp.get()
        all_conf = Variable(self.conf)
        all_conf_mean = Variable(all_conf.mean())
        conf = all_conf / all_conf_mean
        conf.clamp_(0, 2)
        with torch.no_grad():
            self.conf = conf
        self.conf.requires_grad = True
        idxes = np.random.randint(low=0, high=all_states_exp.shape[0], size=batch_size)
        return (
            all_states_exp[idxes],
            all_actions_exp[idxes],
            all_dones_exp[idxes],
            all_next_states_exp[idxes],
            self.conf[idxes]
        )

    def update(self, writer):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # ---Update the discriminator for step 1
            # Samples from current policy's trajectories.
            states, _, _, dones, log_pis, next_states = self.buffer.sample(self.batch_size)

            # Samples from expert's demonstrations.
            states_exp, actions_exp, dones_exp, next_states_exp, conf = self.sample_exp(self.batch_size)

            # Calculate log probabilities of expert actions.
            with torch.no_grad():
                log_pis_exp = self.actor.evaluate_log_pi(states_exp, actions_exp)

            # Update discriminator (retain grad).
            self.update_disc_retain_grad(
                states, dones, log_pis, next_states, states_exp,
                dones_exp, log_pis_exp, next_states_exp, conf
            )

            # ---Update confidence
            self.learning_steps_conf += 1

            # Sample trajectories from our policy.
            states_traj, actions_traj, rewards_traj, next_states_traj = self.buffer.sample_traj(self.traj_batch_size)

            # Update conf
            self.update_conf(states_traj, rewards_traj, writer)

            # ---Update the discriminator for step 2
            # Samples from current policy's trajectories.
            states, _, _, dones, log_pis, next_states = self.buffer.sample(self.batch_size)

            # Samples from expert's demonstrations.
            states_exp, actions_exp, dones_exp, next_states_exp, conf = self.sample_exp(self.batch_size)

            # Calculate log probabilities of expert actions.
            with torch.no_grad():
                log_pis_exp = self.actor.evaluate_log_pi(states_exp, actions_exp)

            # Update discriminator.
            self.update_disc(
                states, dones, log_pis, next_states, states_exp,
                dones_exp, log_pis_exp, next_states_exp, Variable(conf), writer
            )

        # We don't use reward signals here,
        states, actions, _, dones, log_pis, next_states = self.buffer.get()

        # Calculate rewards.
        rewards = self.disc.calculate_reward(states, dones, log_pis, next_states)

        # Update PPO using estimated rewards.
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states, writer)

    def update_disc_retain_grad(self, states, dones, log_pis, next_states,
                                states_exp, dones_exp, log_pis_exp, next_states_exp, conf):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(states_exp, dones_exp, log_pis_exp, next_states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [\frac{r}{\alpha}log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -(F.logsigmoid(logits_exp).mul(conf)).mean()
        loss_disc = loss_pi + loss_exp

        loss_grad = torch.autograd.grad(loss_disc,
                                        self.disc.parameters(),
                                        create_graph=True,
                                        retain_graph=True)
        discLoss_wrt_omega = parameters_to_vector(loss_grad)
        disc_param_vector = parameters_to_vector(self.disc.parameters()).clone().detach()
        disc_param_vector -= self.lr_disc * discLoss_wrt_omega
        self.detached_disc.set_parameters(disc_param_vector)

    def update_disc(self, states, dones, log_pis, next_states,
                    states_exp, dones_exp, log_pis_exp, next_states_exp,
                    conf, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(states_exp, dones_exp, log_pis_exp, next_states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -(F.logsigmoid(logits_exp).mul(conf)).mean()
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

    def update_conf(self, states_traj, rewards_traj, writer):
        learned_rewards_traj = []
        for i in range(len(states_traj)):
            learned_rewards_traj.append(self.detached_disc.g(states_traj[i]).sum().unsqueeze(0))
        outer_loss = self.ranking_loss(rewards_traj, torch.cat(learned_rewards_traj, dim=0))

        outer_loss.backward()
        with torch.no_grad():
            self.conf -= self.lr_conf * self.conf.grad
        self.conf.requires_grad = True
        self.conf.grad.zero_()

        if self.learning_steps_conf % self.epoch_conf == 0:
            writer.add_scalar(
                'loss/outer', outer_loss.item(), self.learning_steps
            )

            # Samples from expert's demonstrations.
            all_states_exp, all_actions_exp, _, all_dones_exp, all_next_states_exp = \
                self.buffer_exp.get()
            all_conf = self.conf
            all_conf_mean = all_conf.mean()

            writer.add_scalar(
                'confidence/mean', all_conf_mean.item(), self.learning_steps
            )

    def ranking_loss(self, truth, approx):
        """
        Calculate the total ranking loss of two list of rewards

        :param truth: list, ground truth rewards of trajectories
        :param approx: torch.Tensor, learned rewards of trajectories
        :param device: cpu or cuda
        :return: ranking loss
        """
        loss_func = nn.MarginRankingLoss().to(self.device)
        loss = torch.Tensor([0]).to(self.device)

        # loop over all the combinations of the rewards
        for c in itertools.combinations(range(approx.shape[0]), 2):
            if truth[c[0]] > truth[c[1]]:
                y = torch.Tensor([1]).to(self.device)
            else:
                y = torch.Tensor([-1]).to(self.device)
            loss += loss_func(approx[c[0]].unsqueeze(0), approx[c[1]].unsqueeze(0), y)
        return loss

    def save_models(self, save_dir):
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        torch.save(self.disc.state_dict(), f'{save_dir}/disc.pkl')
        torch.save(self.actor.state_dict(), f'{save_dir}/actor.pkl')
        all_states_exp, all_actions_exp, _, _, _ = self.buffer_exp.get()
        all_conf = self.conf
        with open(f'{save_dir}/conf.csv', "a") as f:
            for i in range(all_conf.shape[0]):
                f.write(f'{all_conf[i].item()}\n')
