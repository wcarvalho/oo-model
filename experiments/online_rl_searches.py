from ray import tune

def get(search: str = '', agent: str = ''):
  if search == 'sanity_check':
    space = [
        {
          "seed": tune.grid_search([1, 2]),
          "tasks_file": tune.grid_search(['pickup']),
        }
    ]
  else:
    raise NotImplementedError(search)


  return space