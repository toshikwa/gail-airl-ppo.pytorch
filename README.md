# GAIL-PPO in PyTorch
This is a PyTorch implementation of Generative Adversarial Imitation Learning(GAIL)[[1]](#references) based on Proximal Policy Optimization(PPO)[[2]](#references). I tried to make it easy for readers to understand the algorithm. Please let me know if you have any questions.

## Setup
You can install Python liblaries using `pip install -r requirements.txt`. Note that you need a MuJoCo license. Please follow the instruction in [mujoco-py](https://github.com/openai/mujoco-py
) for help.

## References
[[1]](http://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning) Ho, Jonathan, and Stefano Ermon. "Generative adversarial imitation learning." Advances in neural information processing systems. 2016.

[[2]](https://arxiv.org/abs/1707.06347) Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
