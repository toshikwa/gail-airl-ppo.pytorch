# GAIL-PPO in PyTorch
This is a PyTorch implementation of Generative Adversarial Imitation Learning(GAIL)[[1]](#references) based on Proximal Policy Optimization(PPO)[[2]](#references). I tried to make it easy for readers to understand the algorithm. Please let me know if you have any questions.

## Setup
You can install Python liblaries using `pip install -r requirements.txt`. Note that you need a MuJoCo license. Please follow the instruction in [mujoco-py](https://github.com/openai/mujoco-py
) for help.

## Example

### Train your expert
You can train your own expert using Soft Actor-Critic(SAC)[[3,4]](#references). You can see the benchmark of my implementation at [simple-rl](https://github.com/ku2482/simple-rl.pytorch) repo. Also, I've prepared some expert's weights [here](https://github.com/ku2482/gail-ppo.pytorch/tree/master/weights). Please use them if you're only interested in the experiments ahead.

```bash
python train_sac.py --cuda --env_id HalfCheetah-v3 --num_steps 3000000 --seed 0
```

### Collect demonstrations
You need to collect demonstraions using trained expert's weight. Note that `--std` specifies the standard deviation of the gaussian noise add to the action, and `--p_rand` specifies the probability the expert acts randomly.

```bash
python collect_demo.py \
    --weight weights/HalfCheetah-v3.pth --cuda --env_id HalfCheetah-v3 \
    --seed 0 --buffer_size 1000000 --std 0.0 --p_rand 0.0
```

### Train GAIL
Once you prepare demonstrations, you can train GAIL using them. Note that GAIL doesn't use reward signals of the demonstration at all.

```bash
python train_gail.py \
    --buffer buffers/HalfCheetah-v3/1000000_std0.0_prand0.0.pth
    --cuda --env_id HalfCheetah-v3 --num_steps 25000000 --seed 0
```

### Train BCQ
Once you prepare demonstrations, you can train BCQ using them.

```bash
python train_bcq.py \
    --buffer buffers/HalfCheetah-v3/1000000_std0.0_prand0.0.pth
    --cuda --env_id HalfCheetah-v3 --num_steps 100000 --seed 0
```

## References
[[1]](http://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning) Ho, Jonathan, and Stefano Ermon. "Generative adversarial imitation learning." Advances in neural information processing systems. 2016.

[[2]](https://arxiv.org/abs/1707.06347) Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).

[[3]](https://arxiv.org/abs/1801.01290) Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." arXiv preprint arXiv:1801.01290 (2018).

[[4]](https://arxiv.org/abs/1812.05905) Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).