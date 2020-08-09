from abc import ABC, abstractmethod
import os
import numpy as np
import torch


class Algorithm(ABC):

    def __init__(self, state_shape, action_shape, device, seed, gamma):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.learning_steps = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma

    @abstractmethod
    def step(self, env, state, t, step):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
