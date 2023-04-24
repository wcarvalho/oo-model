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
  elif search == 'speed1':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['muzero']),
          "tasks_file": tune.grid_search(['place_split_medium',]),
          "room_size": tune.grid_search([5]),
          "num_dists": tune.grid_search([1]),
          "batch_size": tune.grid_search([32]),
          "num_simulations": tune.grid_search([2, 50]),
          "max_sim_depth": tune.grid_search([10, None]),
          "model_learn_prob": tune.grid_search([0.0, .5, 1.0]),
          "group": tune.grid_search([f'speed_v5']),
        }
    ]
  elif search == 'compare1':
    space = [
        {
          "seed": tune.grid_search([1,2]),
          "agent": tune.grid_search(['r2d2', 'muzero']),
          "tasks_file": tune.grid_search(['pickup']),
          "room_size": tune.grid_search([5]),
          "num_dists": tune.grid_search([1]),
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
            "num_steps": tune.grid_search([5e6]),
            "tasks_file": tune.grid_search(['place_split_easy',
                                            'place_split_medium',
                                            'place_split_hard']),
            "agent": tune.grid_search(['muzero', 'r2d2']),
            "room_size": tune.grid_search([5]),
            "num_dists": tune.grid_search([1]),
            "group": tune.grid_search([f'place_v3']),
            "batch_size": tune.grid_search([32]),
            # "samples_per_insert": tune.grid_search([25, 50]),
            # "model_learn_prob": tune.grid_search([.5, 1.0]),
            # "num_simulations": tune.grid_search([2, 50]),
            # "gumbel_scale": tune.grid_search([1., 0.1]),
        },
    ]
  elif search == 'factored1':
    space = [
        {
            "seed": tune.grid_search([1]),
            "num_steps": tune.grid_search([5e6]),
            "room_size": tune.grid_search([5]),
            "num_dists": tune.grid_search([1]),
            "agent": tune.grid_search(['factored']),
            "group": tune.grid_search([f'speed_v10_factored']),
            "tasks_file": tune.grid_search(['place_split_medium']),
            "batch_size": tune.grid_search([32, 64]),
            "pred_gate": tune.grid_search(['gru', 'sum']),
            "transition_blocks": tune.grid_search([2, 4]),
            "pred_input_selection": tune.grid_search(['attention']),
        },
    ]
  elif search == 'factored2':
    space = [
        {
            "seed": tune.grid_search([1]),
            "num_steps": tune.grid_search([5e6]),
            "room_size": tune.grid_search([5]),
            "num_dists": tune.grid_search([1]),
            "agent": tune.grid_search(['factored']),
            "group": tune.grid_search([f'speed_v10_factored']),
            "tasks_file": tune.grid_search(['place_split_medium']),
            "batch_size": tune.grid_search([32, 64]),
            # "slot_iterations": tune.grid_search([1, 2]),
            "transition_blocks": tune.grid_search([2, 4, 6, 8]),
            # "pred_gate": tune.grid_search(['gru', 'sigtanh', 'sum']),
            "pred_input_selection": tune.grid_search(['task_query']),
        },
    ]
  elif search == 'factored3':
    space = [
        {
            "seed": tune.grid_search([1]),
            "num_steps": tune.grid_search([5e6]),
            "room_size": tune.grid_search([5]),
            "num_dists": tune.grid_search([1]),
            "agent": tune.grid_search(['factored']),
            "group": tune.grid_search([f'speed_v11_factored']),
            "tasks_file": tune.grid_search(['place_split_medium']),
            "batch_size": tune.grid_search([32]),
            "savi_iterations": tune.grid_search([2]),
            "transition_blocks": tune.grid_search([2, 4, 6, 8]),
            # "pred_gate": tune.grid_search(['gru', 'sigtanh', 'sum']),
            "pred_input_selection": tune.grid_search(['task_query']),
        },
    ]
  else:
    raise NotImplementedError(search)


  return space