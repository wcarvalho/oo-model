from ray import tune

def get(search: str = '', agent: str = ''):
  if search == 'sanity_check':
    space = [
        {
          "seed": tune.grid_search([4]),
          "tasks_file": tune.grid_search(['pickup']),
          "importance_sampling_exponent": tune.grid_search([0]),
          "burn_in_length": tune.grid_search([4]),
          "trace_length": tune.grid_search([12]),
          "simulation_steps": tune.grid_search([7]),
          "sequence_period": tune.grid_search([10]),
          "batch_size": tune.grid_search([64, 128]),
          "seperate_model_nets": tune.grid_search([False]),
          "action_source": tune.grid_search(['policy']),
          "group": tune.grid_search(['sanity_check5']),
        }
    ]
  elif search == 'sanity_check2':
    space = [
        {
          "seed": tune.grid_search([4]),
          "tasks_file": tune.grid_search(['pickup']),
          "importance_sampling_exponent": tune.grid_search([0]),
          "burn_in_length": tune.grid_search([4]),
          "trace_length": tune.grid_search([12]),
          "simulation_steps": tune.grid_search([7]),
          "sequence_period": tune.grid_search([10]),
          "resnet_transition_dim": tune.grid_search([128, 256]),
          "batch_size": tune.grid_search([64]),
          "seperate_model_nets": tune.grid_search([False]),
          "action_source": tune.grid_search(['policy']),
          "group": tune.grid_search(['sanity_check6']),
        }
      ]
  else:
    raise NotImplementedError(search)


  return space