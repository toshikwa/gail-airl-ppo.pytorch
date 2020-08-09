import os
import argparse
from datetime import datetime
import torch
import gym

from gail_ppo.buffer import SirializedBuffer
from gail_ppo.algo import OFFLINE_ALGOS
from gail_ppo.trainer import OfflineTrainer


def run(args):
    env_test = gym.make(args.env_id)
    buffer_exp = SirializedBuffer(
        path=args.buffer,
        device=torch.device("cuda" if args.cuda else "cpu")
    )

    algo = OFFLINE_ALGOS[args.algo](
        buffer_exp=buffer_exp,
        state_shape=env_test.observation_space.shape,
        action_shape=env_test.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{args.algo}-seed{args.seed}-{time}')

    trainer = OfflineTrainer(
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, required=True)
    p.add_argument('--num_steps', type=int, default=3*10**7)
    p.add_argument('--env_id', type=str, default='HalfCheetahBulletEnv-v0')
    p.add_argument('--algo', type=str, default='gail')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
