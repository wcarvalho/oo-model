from ray import tune
import rlax

def get(search: str = '', agent: str = ''):
  if search == 'test':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['muzero']),
          "tasks_file": tune.grid_search(['pickup']),
          "room_size": tune.grid_search([7]),
          "num_dists": tune.grid_search([2]),
          "label": tune.grid_search(['v1_5']),
        }
    ]
  elif search == 'compare1':
    space = [
        {
          "seed": tune.grid_search([1,2]),
          "agent": tune.grid_search(['r2d2', 'muzero']),
          "tasks_file": tune.grid_search(['pickup']),
          "room_size": tune.grid_search([5]),
          "num_dists": tune.grid_search([0, 2]),
        }
    ]
  elif search == 'debug1':
    space = [
        {
          "seed": tune.grid_search([4]),
          "agent": tune.grid_search(['muzero']),
          "tasks_file": tune.grid_search(['pickup']),
          "room_size": tune.grid_search([5, 7]),
          "num_dists": tune.grid_search([2]),
          "num_sgd_steps_per_step": tune.grid_search([1, 4]),
          "network_fn": tune.grid_search(['babyai']),
          "builder": tune.grid_search(['old']),
          # "loss_unroll": tune.grid_search(['scan']),
          # "scale_grad": tune.grid_search([0.0, 0.5]),
          "loss_fn": tune.grid_search(['new']),
          # "label": tune.grid_search(['v2']),
          "group": tune.grid_search([f'debug_v5_static']),
        }
    ]
  elif search == 'debug2':
    space = [
        {
          "seed": tune.grid_search([2]),
          "agent": tune.grid_search(['muzero']),
          "tasks_file": tune.grid_search(['pickup']),
          "room_size": tune.grid_search([7]),
          "num_dists": tune.grid_search([2]),
          "network_fn": tune.grid_search(['babyai', 'old_babyai']),
          "builder": tune.grid_search(['old']),
          "loss_fn": tune.grid_search(['new']),
          # "label": tune.grid_search(['v2']),
          "group": tune.grid_search([f'debug_v4']),
        }
    ]
  elif search == 'r2d21':
    space = [
        {
          "seed": tune.grid_search([2,3]),
          "tasks_file": tune.grid_search(['pickup']),
          "agent": tune.grid_search(['r2d2']),
          "room_size": tune.grid_search([7]),
          "num_dists": tune.grid_search([2]),
          "samples_per_insert": tune.grid_search([4.0, 6.0, 10.0, 50.0]),
          "group": tune.grid_search([f'pickup_v17']),
        }
    ]
  elif search == 'muzero1':
    space = [
        {
            "seed": tune.grid_search([1]),
            "tasks_file": tune.grid_search(['pickup']),
            "agent": tune.grid_search(['muzero']),
            "model_coef": tune.grid_search([10.0]),
            # "root_policy_coef": tune.grid_search([1.25, 1.5, 2.0, 2.5, 3.75, 5.0, 7.5, 10.0]),
            "root_policy_coef": tune.grid_search([2.5, 5.0, 10.0, 15.0]),
            "v_target_source": tune.grid_search(['reanalyze']),
            "reanalyze_ratio": tune.grid_search([.5, .9]),
            "group": tune.grid_search([f'pickup_v21']),
        },
    ]

  elif search == 'muzero2':
    space = [
        {
            "seed": tune.grid_search([1,2,3]),
            "tasks_file": tune.grid_search(['pickup']),
            "agent": tune.grid_search(['muzero']),
            "model_coef": tune.grid_search([10.0]),
            "v_target_source": tune.grid_search(['reanalyze']),
            "reanalyze_ratio": tune.grid_search([.5, .75, .9]),
            "group": tune.grid_search([f'pickup_v20']),
        },
    ]

  elif search == 'muzero3':
    space = [
        {
            "seed": tune.grid_search([1]),
            "tasks_file": tune.grid_search(['pickup']),
            "agent": tune.grid_search(['muzero']),
            "group": tune.grid_search([f'pickup_v18']),
        },
    ]
  elif search == 'factored1':
    space = [
        {
            "seed": tune.grid_search([1,2]),
            "num_steps": tune.grid_search([1e6]),
            "tasks_file": tune.grid_search(['pickup']),
            "room_size": tune.grid_search([5]),
            "num_dists": tune.grid_search([0]),
            "tasks_file": tune.grid_search(['pickup']),
            "agent": tune.grid_search(['factored']),
            "gru_init": tune.grid_search(['orthogonal', 'default']),
            "w_init_attn": tune.grid_search([2.0, 1.0]),
            # "transition_blocks": tune.grid_search([2, 4]),
            "group": tune.grid_search([f'pickup_factored_v4_gru']),
            "label": tune.grid_search([f'fixed']),
        },
    ]
  elif search == 'factored2':
    space = [
        {
            "seed": tune.grid_search([1]),
            "num_steps": tune.grid_search([1e6]),
            "tasks_file": tune.grid_search(['pickup']),
            "room_size": tune.grid_search([5]),
            "num_dists": tune.grid_search([0]),
            "tasks_file": tune.grid_search(['pickup']),
            "agent": tune.grid_search(['factored']),
            "gating": tune.grid_search(['sigtanh', 'gru', 'sum']),
            "w_attn_head": tune.grid_search([True, False]),
            "pre_norm": tune.grid_search([True, False]),
            "pred_head": tune.grid_search(['muzero']),
            "group": tune.grid_search([f'pickup_factored_v4_gate']),

        },
    ]
  elif search == 'factored3':
    space = [
        {
            "seed": tune.grid_search([1]),
            "num_steps": tune.grid_search([1e6]),
            "tasks_file": tune.grid_search(['pickup']),
            "room_size": tune.grid_search([5]),
            "num_dists": tune.grid_search([0]),
            "tasks_file": tune.grid_search(['pickup']),
            "agent": tune.grid_search(['factored']),
            "gating": tune.grid_search(['sigtanh', 'gru', 'sum']),
            "w_attn_head": tune.grid_search([True, False]),
            "pre_norm": tune.grid_search([True, False]),
            "pred_head": tune.grid_search(['haiku_mlp']),
            "group": tune.grid_search([f'pickup_factored_v4_gate_mlp']),

        },
    ]
  else:
    raise NotImplementedError(search)


  return space