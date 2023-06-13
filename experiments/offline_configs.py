import dataclasses
import r2d2
from muzero import config as muzero_config
from factored_muzero import config as factored_muzero_config

@dataclasses.dataclass
class R2D2Config(r2d2.R2D2Config):
  discount: float = 0.99


@dataclasses.dataclass
class MuZeroConfig(R2D2Config, muzero_config.MuZeroConfig):
  discount: float = 0.99
  v_target_source = 'return'
  action_source: str = 'value'  # 'policy', 'value', 'mcts'
  target_update_period: int = 100
  batch_size: int = 64
  learning_rate: float = 1e-4
  warmup_steps: int = 1_000
  show_gradients: int = 2500
  behavior_clone: bool = True

@dataclasses.dataclass
class FactoredMuZeroConfig(R2D2Config, factored_muzero_config.FactoredMuZeroConfig):
  discount: float = 0.99
