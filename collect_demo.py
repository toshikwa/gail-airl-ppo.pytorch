import os
import argparse
import torch

from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import SACExpert
from gail_airl_ppo.utils import collect_demo


def run(args):
    env = make_env(args.env_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algo = SACExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        path=args.weight
    )

    buffer, mean_return = collect_demo(
        env=env,
        algo=algo,
        buffer_size=args.buffer_size,
        device=device,
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed
    )

    if os.path.exists(os.path.join(
        'buffers',
        args.env_id,
        f'size{args.buffer_size}_reward{round(mean_return, 2)}.pth'
    )):
        print('Error: demonstrations with the same reward exists')
    else:
        buffer.save(os.path.join(
            'buffers',
            args.env_id,
            f'size{args.buffer_size}_reward{round(mean_return, 2)}.pth'
        ))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weight', type=str, required=True)
    p.add_argument('--env_id', type=str, default='Reacher-v2')
    p.add_argument('--buffer_size', type=int, default=5000)
    p.add_argument('--std', type=float, default=0.0)
    p.add_argument('--p_rand', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
