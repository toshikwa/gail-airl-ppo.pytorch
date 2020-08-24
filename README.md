# GAIL and AIRL in PyTorch
This is a PyTorch implementation of Generative Adversarial Imitation Learning(GAIL)[[1]](#references) and Adversarial Inverse Reinforcement Learning(AIRL)[[2]](#references) based on PPO[[3]](#references). I tried to make it easy for readers to understand the algorithm. Please let me know if you have any questions.


## Setup
You can install Python liblaries using `pip install -r requirements.txt`. Note that you need a MuJoCo license. Please follow the instruction in [mujoco-py](https://github.com/openai/mujoco-py
) for help.

## Example

### Train expert
You can train experts using Soft Actor-Critic(SAC)[[4,5]](#references). We set `num_steps` to 100000 for `InvertedPendulum-v2` and 1000000 for `Hopper-v3`. Also, I've prepared the expert's weights [here](https://github.com/ku2482/gail-ppo.pytorch/tree/master/weights). Please use them if you're only interested in the experiments ahead.

```bash
python train_expert.py --cuda --env_id InvertedPendulum-v2 --num_steps 100000 --seed 0
```

### Collect demonstrations
You need to collect demonstraions using trained expert's weight. Note that `--std` specifies the standard deviation of the gaussian noise add to the action, and `--p_rand` specifies the probability the expert acts randomly. We set `std` to 0.01 not to collect too similar trajectories.

```bash
python collect_demo.py \
    --cuda --env_id InvertedPendulum-v2 \
    --weight weights/InvertedPendulum-v2.pth \
    --buffer_size 1000000 --std 0.01 --p_rand 0.0 --seed 0
```

Mean returns of experts we use in the experiments are listed below.

| Weight(Env) | std | p_rand | Mean Return(without noise) |
|:------------|:---:|:------:|:-----------:|
| InvertedPendulum-v2.pth | 0.01 | 0.0 | 1000(1000)  |
| Hopper-v3.pth | 0.01 | 0.0 | 2534(2791) |


### Train Imitation Learning
You can train IL using demonstrations. We set `rollout_length` to 2000 for `InvertedPendulum-v2` and 50000 for `Hopper-v3`.

```bash
python train_imitation.py \
    --algo gail --cuda --env_id InvertedPendulum-v2 \
    --buffer buffers/InvertedPendulum-v2/size1000000_std0.01_prand0.0.pth \
    --num_steps 100000 --eval_interval 5000 --rollout_length 2000 --seed 0
```

<img src="https://user-images.githubusercontent.com/37267851/91002942-dec90980-e60a-11ea-9bb4-3b5c308bc388.png" title="InvertedPendulum-v2" width=400> <img src="https://user-images.githubusercontent.com/37267851/91002939-dd97dc80-e60a-11ea-940f-6d340306582c.png" title="Hopper-v3" width=400>


## References
[[1]](http://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning) Ho, Jonathan, and Stefano Ermon. "Generative adversarial imitation learning." Advances in neural information processing systems. 2016.

[[2]](https://arxiv.org/abs/1710.11248) Fu, Justin, Katie Luo, and Sergey Levine. "Learning robust rewards with adversarial inverse reinforcement learning." arXiv preprint arXiv:1710.11248 (2017).

[[3]](https://arxiv.org/abs/1707.06347) Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).

[[4]](https://arxiv.org/abs/1801.01290) Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." arXiv preprint arXiv:1801.01290 (2018).

[[5]](https://arxiv.org/abs/1812.05905) Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).
