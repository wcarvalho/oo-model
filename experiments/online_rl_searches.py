from ray import tune

def get(search: str = '', agent: str = ''):
  if search == 'sanity':
    space = [
        {
          "seed": tune.grid_search([4]),
          "tasks_file": tune.grid_search(['pickup']),
          "importance_sampling_exponent": tune.grid_search([0]),
          "burn_in_length": tune.grid_search([0]),
          "simulation_steps": tune.grid_search([8]),
          # "reward_mlps": tune.grid_search([(128, 32), (32,)]),
          # "vpi_mlps": tune.grid_search([(128, 32), (32,)]),
          "action_source": tune.grid_search(['value']),
          "variable_update_period": tune.grid_search([25, 50, 100, 200]),
          # "trace_length": tune.grid_search([12]),
          # "num_blocks": tune.grid_search([4, 8]),
          "batch_size": tune.grid_search([128]),
          # "seperate_model_nets": tune.grid_search([False]),
          # "action_source": tune.grid_search(['policy']),
          "group": tune.grid_search(['sanity_check5']),
        }
    ]
  elif search == 'sanity2':
    space = [
        {
          "seed": tune.grid_search([4]),
          "tasks_file": tune.grid_search(['pickup']),
          "importance_sampling_exponent": tune.grid_search([0]),
          "burn_in_length": tune.grid_search([0]),
          "simulation_steps": tune.grid_search([7]),
          "variable_update_period": tune.grid_search([200, 400]),
          "action_source": tune.grid_search(['value']),
          "group": tune.grid_search(['sanity_check3']),
        }
      ]
  elif search == 'sanity3':
    space = [
        {
          "seed": tune.grid_search([4]),
          "tasks_file": tune.grid_search(['pickup']),
          "importance_sampling_exponent": tune.grid_search([0]),
          "burn_in_length": tune.grid_search([0]),
          "simulation_steps": tune.grid_search([8]),
          # "reward_mlps": tune.grid_search([(128, 32), (32,)]),
          # "vpi_mlps": tune.grid_search([(128, 32), (32,)]),
          "variable_update_period": tune.grid_search([25, 50, 100, 200]),
          "warmup_steps": tune.grid_search([0]),
          # "num_blocks": tune.grid_search([4, 8]),
          # "batch_size": tune.grid_search([64, 128]),
          # "seperate_model_nets": tune.grid_search([False]),
          # "action_source": tune.grid_search(['policy']),
          "group": tune.grid_search(['sanity_check4']),
        }
    ]
  else:
    raise NotImplementedError(search)


  return space