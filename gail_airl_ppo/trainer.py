import os
from time import time, sleep
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, env, env_test, algo, log_dir, seed=0, num_steps=10**5,
                 eval_interval=10**3, num_eval_episodes=5):
        super().__init__()

        # Env to collect samples.
        self.env = env
        self.env.seed(seed)

        # Env for evaluation.
        self.env_test = env_test
        self.env_test.seed(2**31-seed)

        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()

        for step in range(1, self.num_steps + 1):
            # Pass to the algorithm to update state and episode timestep.
            state, t = self.algo.step(self.env, state, t, step)

            # Update the algorithm whenever ready.
            if self.algo.is_update(step):
                self.algo.update(self.writer)

            # Evaluate regularly.
            if step % self.eval_interval == 0:
                self.evaluate(step)
                self.algo.save_models(
                    os.path.join(self.model_dir, f'step{step}'))

        # Wait for the logging to be finished.
        sleep(10)

    def evaluate(self, step):
        mean_return = 0.0

        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            episode_return = 0.0
            done = False

            while (not done):
                action = self.algo.exploit(state)
                state, reward, done, _ = self.env_test.step(action)
                episode_return += reward

            mean_return += episode_return / self.num_eval_episodes

        self.writer.add_scalar('return/test', mean_return, step)
        print(f'Num steps: {step:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
