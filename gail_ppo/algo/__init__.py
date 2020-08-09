from .ppo import PPO
from .sac import SAC
from .gail import GAIL

ONLINE_ALGOS = {
    'ppo': PPO,
    'sac': SAC
}
OFFLINE_ALGOS = {
    'gail': GAIL
}