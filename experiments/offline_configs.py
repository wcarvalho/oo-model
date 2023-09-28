import dataclasses
import r2d2
from muzero import config as muzero_config
from factored_muzero import config as factored_muzero_config
from experiments.config_utils import update_config

@dataclasses.dataclass
class R2D2Config(r2d2.R2D2Config):
  discount: float = 0.99


@dataclasses.dataclass
class MuZeroUpdates:
  num_learner_steps: int = 1e5

  discount: float = 0.99
  v_target_source: str = 'return'
  action_source: str = 'policy'  # 'policy', 'value', 'mcts'
  # target_update_period: int = 100
  batch_size: int = 64
  learning_rate: float = 1e-4
  warmup_steps: int = 10_000
  lr_transition_steps: int = 100_000
  show_gradients: int = 2500
  behavior_clone: bool = True
  mask_model: bool = False

  num_bins: int = 81
  max_scalar_value: float = 2.0

  root_policy_coef: float = 1.0
  root_value_coef: float = 0.25
  model_value_coef: float = 0.25
  model_policy_coef: float = 1.0
  model_reward_coef: float = 1.0


def MuZeroConfig():
  config = muzero_config.MuZeroConfig()
  updates = MuZeroUpdates().__dict__
  update_config(config, strict=False, **updates)
  return config

@dataclasses.dataclass
class FactoredMuZeroUpdates:
  num_sgd_steps_per_step: int = 4

  discount: float = 0.99
  # vision_torso: str = 'babyai'
  transition_blocks: int = 4
  prediction_blocks: int = 2
  mask_model: bool = False
  weight_decay_fn: str = "default"
  weight_decay: float = 1e-4  # very few params


def FactoredMuZeroConfig():
  config = factored_muzero_config.FactoredMuZeroConfig()
  updates = MuZeroUpdates().__dict__
  updates.update(FactoredMuZeroUpdates().__dict__)

  update_config(config, strict=False, **updates)

  return config