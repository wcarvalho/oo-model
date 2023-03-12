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
  elif search == 'muzero_task_model_room5':
    space = [
        {
          "seed": tune.grid_search([2]),
          "tasks_file": tune.grid_search(['pickup', 'pickup_sanity']),
          "room_size": tune.grid_search([5]),
          "trace_length": tune.grid_search([20]),
          "model_combine_state_task": tune.grid_search(['add']),
          "batch_size": tune.grid_search([32]),
          "variable_update_period": tune.grid_search([750, 1000]),
        }
    ]
  elif search == 'r2d2_room5':
    space = [
        {
          "seed": tune.grid_search([2]),
          "tasks_file": tune.grid_search(['pickup', 'pickup_sanity']),
          "agent": tune.grid_search(['r2d2']),
          "room_size": tune.grid_search([5]),
          "trace_length": tune.grid_search([20]),
          "samples_per_insert": tune.grid_search([4.0]),
        }
    ]
  elif search == 'dqn6':
    space = [
        {
          "seed": tune.grid_search([2]),
          "tasks_file": tune.grid_search(['pickup']),
          "room_size": tune.grid_search([5]),
          "v_target_source": tune.grid_search(['q_learning']),
          "action_source": tune.grid_search(['value']),
          "samples_per_insert": tune.grid_search([4.0, 50.0]),
          "policy_coef": tune.grid_search([0.0]),
          "model_coef": tune.grid_search([0.0]),
          "ema_update": tune.grid_search([0.0]),
          "value_coef": tune.grid_search([1.0]),
          "scalar_step_size": tune.grid_search([.25, 1.0]),
          "max_scalar_value": tune.grid_search([50.0]),
        }
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