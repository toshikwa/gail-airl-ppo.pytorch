from .ppo import PPO
from .sac import SAC, SACExpert
from .gail import GAIL

ONLINE_ALGOS = {
    'ppo': PPO,
    'sac': SAC
}
OFFLINE_ALGOS = {
    'gail': GAIL
}
EXPERT_ALGOS = {
    'sac': SACExpert
}