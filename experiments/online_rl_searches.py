from ray import tune
import rlax

def get(search: str = '', agent: str = ''):
  if search == 'ema_step_size':
    space = [
        {
          "seed": tune.grid_search([2]),
          "tasks_file": tune.grid_search(['pickup']),
          "room_size": tune.grid_search([5]),
          "v_target_source": tune.grid_search(['q_learning']),
          "action_source": tune.grid_search(['value']),
          "policy_coef": tune.grid_search([0.0]),
          "ema_update": tune.grid_search([0.0, 0.1, 0.01, 0.001]),
          "value_coef": tune.grid_search([1.0]),
          "step_size": tune.grid_search([.25]),
          "max_value": tune.grid_search([10.0]),
        }
    ]
  elif search == 'long2':
    space = [
        {
          "seed": tune.grid_search([2]),
          "tasks_file": tune.grid_search(['long', 'pickup_place']),
          "agent": tune.grid_search(['r2d2', 'muzero']),
          "room_size": tune.grid_search([8]),
          "num_dists": tune.grid_search([2]),
          "num_steps": tune.grid_search([30e6]),
          # "samples_per_insert": tune.grid_search([6.0, 10.0]),
          # "discount": tune.grid_search([.997]),
          # "sentence_dim": tune.grid_search([64]),
          # "word_dim": tune.grid_search([64]),
          # "task_dim": tune.grid_search([0, 64]),
          # "trace_length": tune.grid_search([20, 40]),
          # "adam_eps": tune.grid_search([1e-8, 1e-3]),
        }
    ]
  elif search == 'muzero_task_model_room8':
    space = [
        {
          "seed": tune.grid_search([2]),
          "tasks_file": tune.grid_search(['place']),
          "agent": tune.grid_search(['muzero']),
          "room_size": tune.grid_search([5]),
          "num_dists": tune.grid_search([2]),
          # "trace_length": tune.grid_search([20]),
          # "model_combine_state_task": tune.grid_search(['add_state_bias',
          #                                               'add_head', 'add_head_bias',]),
          "adam_eps": tune.grid_search([1e-8, 1e-3]),
          # "variable_update_period": tune.grid_search([750, 1000, 1250, 1500]),
        }
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