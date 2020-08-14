import torch
from torch import nn

from .utils import build_mlp


class Noise(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(400, 300),
                 hidden_activation=nn.ReLU(inplace=True), std=0.05):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            output_activation=nn.Tanh()
        )
        self.std = std

    def forward(self, states, actions):
        noises = self.net(torch.cat([states, actions], dim=1))
        return actions.add_(self.std * noises).clamp_(-1.0, 1.0)


class CVAE(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(750, 750),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.latent_dim = 2 * action_shape[0]
        self.encoder = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=2 * self.latent_dim,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.decoder = build_mlp(
            input_dim=state_shape[0] + self.latent_dim,
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            output_activation=nn.Tanh()
        )

    def forward(self, states, actions):
        means, log_stds = self._encode(states, actions)
        latents = means + log_stds.exp() * torch.randn_like(log_stds)
        reconsts = self._decode(states, latents)
        return reconsts, means, 2 * log_stds

    def generate(self, states):
        latents = torch.randn(
            (states.size(0), self.latent_dim),
            device=states.device
        ).clamp_(-0.5, 0.5)
        with torch.no_grad():
            return self._decode(states, latents)

    def _encode(self, states, actions):
        xs = torch.cat([states, actions], dim=1)
        means, log_stds = self.encoder(xs).chunk(2, dim=1)
        return means, log_stds.clamp_(-4, 15)

    def _decode(self, states, latents):
        return self.decoder(torch.cat([states, latents], dim=1))
