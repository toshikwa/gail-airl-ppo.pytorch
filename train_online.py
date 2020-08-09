import os
import argparse
from datetime import datetime
import torch
import gym

from gail_ppo.algo import ONLINE_ALGOS
from gail_ppo.trainer import OnlineTrainer


def run(args):
    env = gym.make(args.env_id)
    env_test = gym.make(args.env_id)

    algo = ONLINE_ALGOS[args.algo](
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{args.algo}-seed{args.seed}-{time}')

    trainer = OnlineTrainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=3*10**6)
    p.add_argument('--env_id', type=str, default='HalfCheetah-v3')
    p.add_argument('--algo', type=str, default='sac')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
