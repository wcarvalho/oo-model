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
          "num_dists": tune.grid_search([0, 2]),
          "network_fn": tune.grid_search(['babyai']),
          "builder": tune.grid_search(['old']),
          # "loss_unroll": tune.grid_search(['scan']),
          # "scale_grad": tune.grid_search([0.0, 0.5]),
          "loss_fn": tune.grid_search(['new']),
          # "label": tune.grid_search(['v2']),
          "group": tune.grid_search([f'debug_v4']),
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
  elif search == 'r2d2_pickup6':
    space = [
        {
          "seed": tune.grid_search([1,2,3]),
          "tasks_file": tune.grid_search(['pickup']),
          "agent": tune.grid_search(['r2d2']),
          "room_size": tune.grid_search([7]),
          "num_dists": tune.grid_search([0, 2]),
          "num_steps": tune.grid_search([5e6]),
          "samples_per_insert": tune.grid_search([4.0]),
          "epsilon_min": tune.grid_search([1]),
          "epsilon_max": tune.grid_search([3]),
          "epsilon_base": tune.grid_search([.1]),
        }
    ]
  elif search == 'muzero1':
    space = [
        {
          "seed": tune.grid_search([1]),
          "tasks_file": tune.grid_search(['pickup']),
          "agent": tune.grid_search(['muzero']),
          # "samples_per_insert": tune.grid_search([75, 100]),
          "room_size": tune.grid_search([5]),
          "num_dists": tune.grid_search([2]),
          "adam_eps": tune.grid_search([1e-3, 1e-8]),
          "max_grad_norm": tune.grid_search([5.0, 80.0]),
          # "scale_grad": tune.grid_search([0.0, .5, .9]),
          "group": tune.grid_search([f'pickup_r5_d2_v3']),
        },
    ]

  elif search == 'muzero2':
    space = [
        {
          "seed": tune.grid_search([1]),
          "tasks_file": tune.grid_search(['pickup']),
          "agent": tune.grid_search(['muzero']),
          # "samples_per_insert": tune.grid_search([75, 100]),
          "room_size": tune.grid_search([7]),
          "num_dists": tune.grid_search([2]),
          # "num_steps": tune.grid_search([3e6]),
          # "adam_eps": tune.grid_search([1e-3]),
          # r2d2: seqeunce_lenght=120, period=30
          # muzero: length=20, period=20
          "sequence_period": tune.grid_search([5, 10, 19]),
          "group": tune.grid_search([f'pickup_2_v4_seq']),
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