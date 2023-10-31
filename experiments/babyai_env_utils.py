

import dm_env
import functools
import os.path
import yaml

from acme import wrappers

from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper
from envs.babyai_kitchen.wrappers import RGBImgFullyObsWrapper
from envs.babyai_kitchen.wrappers import MissionIntegerWrapper
from envs.babyai_kitchen.utils import InstructionsPreprocessor

from envs.multitask_kitchen import MultitaskKitchen

def get_kitchen_tasks_file(tasks_file: str):
  """Open BabyAI Kitchen tasks file.
  
  Args:
      tasks_file (str, optional): Description
  
  Returns:
      TYPE: Description
  """
  tasks_file = tasks_file or 'place'
  return f"envs/babyai_kitchen/tasks/{tasks_file}.yaml"

def open_kitchen_tasks_file(tasks_file: str='place', path: str='.'):
  """Open BabyAI Kitchen tasks file.
  
  Args:
      tasks_file (str, optional): Description
  
  Returns:
      TYPE: Description
  """
  tasks_file = get_kitchen_tasks_file(tasks_file)
  tasks_file = os.path.join(path, tasks_file)
  assert os.path.exists(tasks_file), tasks_file

  with open(tasks_file, 'r') as f:
    tasks = yaml.load(f, Loader=yaml.SafeLoader)
  return tasks

def make_kitchen_environment(
  evaluation: bool = False,
  evaluate_train_test: bool = False,
  room_size: int=7,
  num_dists: int=2,
  agent_view_size: int = 7,
  partial_obs: bool = False,
  max_text_length=10,
  tile_size=8,
  path='.',
  tasks_file='',
  debug=False,
  nseeds=0,
  return_gym_env=False,
  wrapper_list=None,
  test_larger=True,
  **kwargs,
  ) -> dm_env.Environment:
  """Loads environments."""

  tasks_file = tasks_file or 'place'
  tasks = open_kitchen_tasks_file(tasks_file, path=path)

  if evaluation and 'test' in tasks:
    task_dicts = tasks['test']
    if evaluate_train_test:
      task_dicts = task_dicts + tasks['train']
  else:
    task_dicts = tasks['train']

  instr_preproc = InstructionsPreprocessor(
    path=os.path.join(path, "envs/babyai_kitchen/vocab.json"))
  env_wrappers = [functools.partial(MissionIntegerWrapper,
        instr_preproc=instr_preproc,
        max_length=max_text_length)]

  if partial_obs:
    # shape: always 56 x 56 x 3
    env_wrappers.append(functools.partial(RGBImgPartialObsWrapper,
      tile_size=tile_size))
  else:
    # room-size=10 --> image: 80 x 80  x 3
    # room-size=8  --> image: 64 x 64  x 3 (conv = 7x7 = 49)
    # room-size=7  --> image: 56 x 56  x 3 (conv = 6x6 = 36)
    # room-size=5  --> image: 40 x 40  x 3 (conv = 5x5 = 25)
    env_wrappers.append(functools.partial(RGBImgFullyObsWrapper,
      tile_size=tile_size))

  nseeds=0 if evaluation else nseeds

  dm_env = MultitaskKitchen(
    task_dicts=task_dicts,
    tasks_file=tasks,
    tile_size=tile_size,
    agent_view_size=agent_view_size,
    path=path,
    num_dists=num_dists,
    room_size=room_size,
    wrappers=env_wrappers,
    debug=debug,
    nseeds=nseeds,
    test_larger=test_larger and evaluation,
    **kwargs
    )


  wrapper_list = wrapper_list or [
    wrappers.ObservationActionRewardWrapper,
    wrappers.SinglePrecisionWrapper,
  ]
  dm_env = wrappers.wrap_all(dm_env, wrapper_list)

  if return_gym_env:
    gym_env = dm_env.env
    return dm_env, gym_env
  else:
    return dm_env
