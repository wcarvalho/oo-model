from ray import tune
import rlax

def get(search: str = '', agent: str = ''):
  if search == 'test':
    space = [
        {
          "seed": tune.grid_search([1,2,3]),
          "agent": tune.grid_search(['r2d2', 'muzero']),
          "tasks_file": tune.grid_search(['pickup']),
          "room_size": tune.grid_search([7]),
          "label": tune.grid_search(['v1_3']),
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
        },
        {
            "seed": tune.grid_search([1,2,3]),
            "tasks_file": tune.grid_search(['pickup']),
            "agent": tune.grid_search(['r2d2']),
            "room_size": tune.grid_search([7]),
            "num_dists": tune.grid_search([0, 2]),
            "num_steps": tune.grid_search([5e6]),
            "samples_per_insert": tune.grid_search([4.0]),
            "num_epsilons": tune.grid_search([10]),
            "epsilon_min": tune.grid_search([1e-2]),
            "epsilon_max": tune.grid_search([1]),
            "epsilon_base": tune.grid_search([.1]),
        }
        # {
        #   "seed": tune.grid_search([2]),
        #   "tasks_file": tune.grid_search(['pickup']),
        #   "agent": tune.grid_search(['r2d2']),
        #   "room_size": tune.grid_search([7]),
        #   "num_dists": tune.grid_search([0, 2]),
        #   "num_steps": tune.grid_search([5e6]),
        #   "samples_per_insert": tune.grid_search([6.0]),
        #   "epsilon_min": tune.grid_search([1]),
        #   "epsilon_max": tune.grid_search([8]),
        #   "epsilon_base": tune.grid_search([.4]),
        # }
    ]
  elif search == 'muzero_pickup6':
    space = [
        {
          "seed": tune.grid_search([1,2,3]),
          "tasks_file": tune.grid_search(['pickup']),
          "agent": tune.grid_search(['muzero']),
          # "samples_per_insert": tune.grid_search([50, 75]),
          "room_size": tune.grid_search([7]),
          "num_dists": tune.grid_search([2]),
          "num_steps": tune.grid_search([3e6]),
          "adam_eps": tune.grid_search([1e-3, 1e-8]),
          "max_grad_norm": tune.grid_search([80.0, 5.0]),
          "group": tune.grid_search([f'pickup_2_v3']),
        },
    ]
  elif search == 'double_check3':
    space = [
        {
          "seed": tune.grid_search([2]),
          "tasks_file": tune.grid_search(['pickup']),
          "agent": tune.grid_search(['muzero', 'r2d2']),
          "room_size": tune.grid_search([7]),
          "num_dists": tune.grid_search([2]),
          "conv_out_dim": tune.grid_search([0, 32]),
          # "batch_size": tune.grid_search([64]),
          # "samples_per_insert": tune.grid_search([50, 75]),
          # "adam_eps": tune.grid_search([1e-3]),
          # "max_grad_norm": tune.grid_search([80.0]),
        },
    ]
  elif search == 'double_check4':
    space = [
        {
          "seed": tune.grid_search([2]),
          "tasks_file": tune.grid_search(['pickup']),
          "agent": tune.grid_search(['muzero']),
          "room_size": tune.grid_search([7]),
          "num_dists": tune.grid_search([2]),
          "vpi_mlps": tune.grid_search([(512,), (128, 32)]),
          "reward_mlps": tune.grid_search([(512,), (32,)]),
          # "batch_size": tune.grid_search([64]),
          # "samples_per_insert": tune.grid_search([50, 75]),
          # "adam_eps": tune.grid_search([1e-3]),
          # "max_grad_norm": tune.grid_search([80.0]),
        },
    ]
  elif search == 'speed2':
    space = [
        {
          "seed": tune.grid_search([2]),
          "tasks_file": tune.grid_search(['pickup']),
          "room_size": tune.grid_search([5]),
          "batch_size": tune.grid_search([64]),
          "samples_per_insert": tune.grid_search([25.0, 50.0, 15.0]),
        },
        # {
        #   "seed": tune.grid_search([2]),
        #   "tasks_file": tune.grid_search(['pickup']),
        #   "room_size": tune.grid_search([5]),
        #   "max_grad_norm": tune.grid_search([0.0]),
        # }
    ]
  elif search == 'pickup2':
    space = [
        {
          "seed": tune.grid_search([2]),
          "tasks_file": tune.grid_search(['pickup']),
          "room_size": tune.grid_search([8]),
          "num_steps": tune.grid_search([5e6]),
          "agent": tune.grid_search(['r2d2', 'muzero']),
          "samples_per_insert": tune.grid_search([50.0]),
        },
        {
          "seed": tune.grid_search([2]),
          "tasks_file": tune.grid_search(['pickup']),
          "room_size": tune.grid_search([8]),
          "num_steps": tune.grid_search([5e6]),
          "agent": tune.grid_search(['r2d2']),
          "samples_per_insert": tune.grid_search([4.0]),
        }
    ]
  elif search == 'place2':
    space = [
        {
          "seed": tune.grid_search([2]),
          "tasks_file": tune.grid_search(['pickup_place']),
          "room_size": tune.grid_search([8]),
          "num_steps": tune.grid_search([10e6]),
          "agent": tune.grid_search(['r2d2', 'muzero']),
          "samples_per_insert": tune.grid_search([50.0]),
        },
        {
          "seed": tune.grid_search([2]),
          "tasks_file": tune.grid_search(['pickup_place']),
          "room_size": tune.grid_search([8]),
          "num_steps": tune.grid_search([10e6]),
          "agent": tune.grid_search(['r2d2']),
          "samples_per_insert": tune.grid_search([4.0]),
        }
    ]

  elif search == 'grads5':
    space = [
        {
          "seed": tune.grid_search([2]),
          "tasks_file": tune.grid_search(['pickup']),
          "room_size": tune.grid_search([5]),
          "v_target_source": tune.grid_search(['q_learning']),
          "action_source": tune.grid_search(['value']),
          "samples_per_insert": tune.grid_search([4.0]),
          "policy_coef": tune.grid_search([0.0]),
          "model_coef": tune.grid_search([0.0]),
          # "ema_update": tune.grid_search([0.0]),
          "show_grads": tune.grid_search([1]),
          # "step_size": tune.grid_search([.25, 1.0]),
          # "max_value": tune.grid_search([10.0, 50.0]),
        }
    ]

  else:
    raise NotImplementedError(search)


  return space