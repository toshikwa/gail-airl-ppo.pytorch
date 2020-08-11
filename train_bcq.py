import os
import argparse
from datetime import datetime
import torch
import gym

from gail_ppo.buffer import SerializedBuffer
from gail_ppo.algo import BCQ
from gail_ppo.trainer import OfflineTrainer


def run(args):
    env_test = gym.make(args.env_id)
    buffer_exp = SerializedBuffer(
        path=args.buffer,
        device=torch.device("cuda" if args.cuda else "cpu")
    )

    algo = BCQ(
        buffer_exp=buffer_exp,
        state_shape=env_test.observation_space.shape,
        action_shape=env_test.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, 'gail', f'seed{args.seed}-{time}')

    trainer = OfflineTrainer(
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, required=True)
    p.add_argument('--num_steps', type=int, default=10**5)
    p.add_argument('--eval_interval', type=int, default=10**3)
    p.add_argument('--env_id', type=str, default='HalfCheetah-v3')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
