import os
import torch

from gail_airl_ppo.buffer import Buffer, SerializedBuffer
from gail_airl_ppo.env import make_env


def mix_demo(buffer_name, size):
    device = torch.device("cpu")
    env_name = buffer_name[0].split('/')[1]
    env = make_env(env_name)

    output_buffer = Buffer(
        buffer_size=sum(size),
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    buffers = []
    for i_buffer, name in enumerate(buffer_name):
        buffers.append(
            SerializedBuffer(
                path=name,
                device=device
            )
        )
        states, actions, rewards, dones, next_states = buffers[i_buffer].get()
        for i_demo in range(size[i_buffer]):
            output_buffer.append(
                states[i_demo].numpy(),
                actions[i_demo].numpy(),
                rewards[i_demo].numpy(),
                dones[i_demo].numpy(),
                next_states[i_demo].numpy()
            )

    rewards_name = ''
    for name in buffer_name:
        mean_reward = name.split('reward')[1].split('.pth')[0]
        rewards_name = rewards_name + '_' + mean_reward

    if os.path.exists(os.path.join(
        'buffers',
        env_name,
        f'size{sum(size)}_reward{rewards_name}.pth'
    )):
        print('Error: demonstrations with the same reward exists')
    else:
        output_buffer.save(os.path.join(
            'buffers',
            env_name,
            f'size{sum(size)}_reward{rewards_name}.pth'
        ))


if __name__ == '__main__':
    BUFFER_NAME = (
        'buffers/Reacher-v2/size5000_reward-4.35.pth',
        'buffers/Reacher-v2/size5000_reward-5.51.pth',
        'buffers/Reacher-v2/size5000_reward-8.85.pth',
        'buffers/Reacher-v2/size5000_reward-44.92.pth',
        'buffers/Reacher-v2/size5000_reward-81.01.pth'
    )
    SIZE = (1000, 1000, 1000, 1000, 1000)
    mix_demo(buffer_name=BUFFER_NAME,
             size=SIZE)
