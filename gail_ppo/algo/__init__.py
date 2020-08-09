from .ppo import PPO
from .sac import SAC, SACExpert
from .gail import GAIL
from .bcq import BCQ

RL_ALGOS = {
    'ppo': PPO,
    'sac': SAC
}
EXPERT_ALGOS = {
    'sac': SACExpert
}