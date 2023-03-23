from ray import tune
import rlax

def get(search: str = '', agent: str = ''):
  if search == 'test':
    space = [
        {
          "seed": tune.grid_search([1,2,3]),
          "agent": tune.grid_search(['muzero']),
          "tasks_file": tune.grid_search(['pickup']),
          "room_size": tune.grid_search([7]),
          "num_dists": tune.grid_search([2]),
          "label": tune.grid_search(['v1_4']),
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
          "seed": tune.grid_search([1,2,3]),
          "tasks_file": tune.grid_search(['pickup']),
          "agent": tune.grid_search(['r2d2']),
          "room_size": tune.grid_search([5, 7]),
          "num_dists": tune.grid_search([2]),
          "samples_per_insert": tune.grid_search([6.0]),
          "group": tune.grid_search([f'debug_v4']),
        }
    ]
  elif search == 'muzero1':
    space = [
        {
          "seed": tune.grid_search([1,2,3]),
          "tasks_file": tune.grid_search(['pickup']),
          "agent": tune.grid_search(['muzero']),
          "room_size": tune.grid_search([7]),
          "num_dists": tune.grid_search([2]),
          "model_coef": tune.grid_search([.5]),
          "v_target_source": tune.grid_search(['reanalyze', 'reanalyze2']),
          "reanalyze_ratio": tune.grid_search([.75, .9]),
          "group": tune.grid_search([f'pickup_d2_v8_reanalyze']),
          # "label": tune.grid_search([f'8']),
        },
    ]

  elif search == 'muzero2':
    space = [
        {
          "seed": tune.grid_search([1,2,3]),
          "tasks_file": tune.grid_search(['pickup']),
          "agent": tune.grid_search(['muzero']),
          "room_size": tune.grid_search([7, 5]),
          "num_dists": tune.grid_search([2]),
          "group": tune.grid_search([f'pickup_d2_v4']),
          "label": tune.grid_search([f'13']),
        },
    ]

  elif search == 'muzero3':
    space = [
        {
          "seed": tune.grid_search([1]),
          "tasks_file": tune.grid_search(['pickup']),
          "agent": tune.grid_search(['muzero']),
          # "samples_per_insert": tune.grid_search([75, 100]),
          "room_size": tune.grid_search([7]),
          "num_dists": tune.grid_search([2]),
          "reward_mlps": tune.grid_search([[], [32]]),
          "vpi_mlps": tune.grid_search([[], [128, 32]]),
          "policy_loss": tune.grid_search(["cross_entropy", "kl_forward", "kl_back"]),
          "group": tune.grid_search([f'policy_loss_2_v2']),
        },
    ]
  else:
    raise NotImplementedError(search)


  return space