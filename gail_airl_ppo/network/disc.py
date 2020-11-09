import torch
from torch import nn
import torch.nn.functional as F

from .utils import build_mlp, build_param_list


class GAILDiscrim(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(100, 100),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))

    def calculate_reward(self, states, actions):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(states, actions))


class AIRLDiscrim(nn.Module):

    def __init__(self, state_shape, gamma,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)):
        super().__init__()

        self.g = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r
        )
        self.h = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        self.gamma = gamma

    def f(self, states, dones, next_states):
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(self, states, dones, log_pis, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(states, dones, next_states) - log_pis

    def calculate_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return -F.logsigmoid(-logits)


class AIRLDetachedDiscrim:

    def __init__(self, state_shape, gamma, device,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)):

        self.g_weights, self.g_biases = build_param_list(
            input_dim=state_shape[0],
            output_dim=1,
            device=device,
            hidden_units=hidden_units_r,
            requires_grad=True,
        )
        self.h_weights, self.h_biases = build_param_list(
            input_dim=state_shape[0],
            output_dim=1,
            device=device,
            hidden_units=hidden_units_v,
            requires_grad=True
        )

        self.hidden_units_r = hidden_units_r
        self.hidden_units_v = hidden_units_v
        self.hidden_activation_r = hidden_activation_r
        self.hidden_activation_v = hidden_activation_v
        self.gamma = gamma

    def g(self, states):
        x = states
        for weight, bias in zip(self.g_weights, self.g_biases):
            x = self.hidden_activation_r(x.mm(weight.transpose(0, 1)) + bias)
        return x

    def h(self, states):
        x = states
        for weight, bias in zip(self.h_weights, self.h_biases):
            x = self.hidden_activation_v(x.mm(weight.transpose(0, 1)) + bias)
        return x

    def f(self, states, dones, next_states):
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(self, states, dones, log_pis, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(states, dones, next_states) - log_pis

    def calculate_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return -F.logsigmoid(-logits)

    def set_parameters(self, vector):
        pointer = 0
        for layer in range(len(self.g_weights)):
            n_param = int(self.g_weights[layer].shape[0] * self.g_weights[layer].shape[1])
            self.g_weights[layer] = vector[pointer: pointer + n_param].view(self.g_weights[layer].shape)
            pointer += n_param
            n_param = self.g_biases[layer].shape[0]
            self.g_biases[layer] = vector[pointer: pointer + n_param].view(self.g_biases[layer].shape)
            pointer += n_param

        for layer in range(len(self.h_weights)):
            n_param = int(self.h_weights[layer].shape[0] * self.h_weights[layer].shape[1])
            self.h_weights[layer] = vector[pointer: pointer + n_param].view(self.h_weights[layer].shape)
            pointer += n_param
            n_param = self.h_biases[layer].shape[0]
            self.h_biases[layer] = vector[pointer: pointer + n_param].view(self.h_biases[layer].shape)
            pointer += n_param

        assert pointer == vector.shape[0]
