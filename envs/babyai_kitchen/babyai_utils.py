import os
import copy
import yaml
import babyai.utils
import babyai.levels.iclr19_levels as iclr19_levels
from envs.babyai_kitchen.levelgen import KitchenLevel
from envs.babyai_kitchen.multilevel import MultiLevel

def load_babyai_env(default_env_kwargs, rootdir='.'):
  # vocab/tasks paths
  vocab_path = os.path.join(rootdir, "preloads/babyai/vocab.json")
  task_path = os.path.join(rootdir, "preloads/babyai/tasks.json")

  # dynamically load environment to use. corresponds to gym envs.
  import babyai.levels.iclr19_levels as iclr19_levels
  level = default_env_kwargs['env']['level']
  env_class = getattr(iclr19_levels, f"Level_{level}")

  instr_preprocessor = load_instr_preprocessor(vocab_path)

  # -----------------------
  # setup kwargs
  # -----------------------
  level_kwargs=default_env_kwargs.get('level', {})
  env_kwargs = eval_env_kwargs = dict(
          instr_preprocessor=instr_preprocessor,
          env_class=env_class,
          level_kwargs=level_kwargs,
          )

  return env_kwargs, eval_env_kwargs

def load_instr_preprocessor(path="preloads/babyai/vocab.json"):
  instr_preprocessor = babyai.utils.format.InstructionsPreprocessor(path=path)

  path = instr_preprocessor.vocab.path
  if not os.path.exists(path):
      raise RuntimeError(f"Please create vocab and put in {path}")
  else:
      print(f"Successfully loaded {path}")

  return instr_preprocessor

def load_babyai_kitchen_env(default_env_kwargs, rootdir='.'):
  # -----------------------
  # load vocabulary
  # -----------------------
  vocab_path = os.path.join(rootdir, "preloads/babyai_kitchen/vocab.json")
  instr_preprocessor = load_instr_preprocessor(vocab_path)
  default_env_kwargs.update(dict(
      instr_preprocessor=instr_preprocessor,
      env_class=KitchenMultiLevel,
      ))

  # -----------------------
  # load file w/ tasks 
  # -----------------------
  tasks_file = default_env_kwargs.get('tasks_file', None)
  with open(os.path.join(rootdir, tasks_file), 'r') as f:
    tasks = yaml.load(f, Loader=yaml.SafeLoader)

  # -----------------------
  # load file w/ sets of objects
  # -----------------------
  sets_file = default_env_kwargs.get('sets_file',
      "tasks/babyai_kitchen/default_sets.yaml")
  with open(os.path.join(rootdir, sets_file), 'r') as f:
    sets = yaml.load(f, Loader=yaml.SafeLoader)

  # -----------------------
  # load kwargs for meta-environment used for multi-level training
  # -----------------------
  env_kwargs = dict(default_env_kwargs)
  eval_env_kwargs = dict(default_env_kwargs)
  level_kwargs = default_env_kwargs.get('level_kwargs', None)

  env_kwargs['level_kwargs'] = constuct_kitchenmultilevel_kwargs(
      task_dicts=tasks['train'],
      level_kwargs=level_kwargs,
      sets=sets)

  eval_env_kwargs['level_kwargs'] = constuct_kitchenmultilevel_kwargs(
      task_dicts=tasks['test'],
      level_kwargs=level_kwargs,
      sets=sets)

  # level_kwargs = next(iter(env_kwargs['level_kwargs'].values()))
  env = MultiLevel(env_kwargs['level_kwargs'])
  import ipdb; ipdb.set_trace()
  return env_kwargs, eval_env_kwargs


def constuct_kitchenmultilevel_kwargs(task_dicts, **kwargs):
  all_level_kwargs = dict()
  unnamed_counter = 0
  for task_dict in task_dicts:
      level_kwargs = construct_kitchenlevel_kwargs(task_dict=task_dict, **kwargs)

      if 'name' in task_dict:
          level_name = task_dict['name']
      else:
          level_name = 'unnamed%d' % unnamed_counter
          unnamed_counter += 1

      all_level_kwargs[level_name] = level_kwargs
  return all_level_kwargs


def construct_kitchenlevel_kwargs(task_dict, level_kwargs=None, sets=None):
  level_kwargs = copy.deepcopy(level_kwargs) or dict()
  task_dict = copy.deepcopy(task_dict) or dict()
  sets = sets or dict()


  def set_items(name):
      if name in sets:
          return sets[name]
      else:
          return [name]
  if 'taskarg_options' in task_dict:
      argops = task_dict['taskarg_options']
      for k, v in argops.items():
          names = argops[k]
          # ======================================================
          # go through each name.
          # names = list:
          #   for each name:
          #       if name points to set, add set values to items list
          #       if not, add name itself
          # names = str:
          #   same logic as for name above
          # ======================================================
          items = []
          if isinstance(names, list):
              for i in names:
                  items.extend(set_items(i))
          elif isinstance(names, str):
              items.extend(set_items(names))
          else:
              raise RuntimeError(str(names))

          argops[k] = items
  
  task_dict.pop("name", None)
  level_kwargs.update(task_dict)

  return level_kwargs

