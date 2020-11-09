from .ppo import PPO
from .sac import SAC, SACExpert
from .gail import GAIL
from .airl import AIRL
from .sil import SIL

ALGOS = {
    'gail': GAIL,
    'airl': AIRL,
    'sil': SIL
}
