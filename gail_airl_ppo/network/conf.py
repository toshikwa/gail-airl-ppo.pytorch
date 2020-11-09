import torch
from torch import nn
import torch.nn.functional as F

from .utils import build_mlp


class SILConfidence(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        return torch.sigmoid(self.net(torch.cat([states, actions], dim=-1)))
