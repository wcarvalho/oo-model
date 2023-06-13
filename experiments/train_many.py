from typing import Optional, Union, List, Dict

from absl import flags

import multiprocessing as mp
import os
import time

from pathlib import Path
from ray import tune
import subprocess

from acme.utils import paths

from experiments import logger as wandb_logger 
from experiments import config_utils

flags.DEFINE_integer('num_actors', 1, 'number of actors.')
flags.DEFINE_integer('num_cpus', 4, 'number of cpus.')
flags.DEFINE_float('num_gpus', 1, 'number of gpus.')
flags.DEFINE_bool('skip', True, 'whether to skip experiments that have already run.')

FLAGS = flags.FLAGS

DEFAULT_LABEL = ''


def make_program_command(
    agent: str,
    folder: str,
    agent_config: str,
    env_config: str,
    wandb_init_kwargs: dict,
    filename: str = '',
    num_actors: int = 2,
    run_distributed: bool = False,
    **kwargs,
):
  wandb_project = wandb_init_kwargs['project']
  wandb_group = wandb_init_kwargs['group']
  wandb_name = wandb_init_kwargs['name']
  wandb_entity = wandb_init_kwargs['entity']
  wandb_dir = wandb_init_kwargs.get("dir", None)

  assert filename, 'please provide file'
  str = f"""python {filename}
		--agent={agent}
		--use_wandb=True
		--wandb_project={wandb_project}
		--wandb_entity={wandb_entity}
		--wandb_group={wandb_group}
    --wandb_name={wandb_name}
    --wandb_dir={wandb_dir}
    --folder={folder}
    --agent_config={agent_config}
    --env_config={env_config}
    --num_actors={num_actors}
    --run_distributed={run_distributed}
    --train_single=True
  """
  for k, v in kwargs:
    str += "--{k}={v}"
  return str


def create_and_run_program(
    config,
    make_program_command,
    root_path: str = '.',
    folder : Optional[str] = None,
    wandb_init_kwargs: dict = None,
    default_env_kwargs: dict = None,
    skip: bool = True,
    debug: bool = False):
  """Create and run launchpad program
  """

  agent = config.pop('agent', 'muzero')
  cuda = config.pop('cuda', None)
  label = config.pop('label', DEFAULT_LABEL)

  # -----------------------
  # update env kwargs with config. HACK
  # -----------------------
  default_env_kwargs = default_env_kwargs or {}
  env_kwargs = dict()
  for key, value in default_env_kwargs.items():
    env_kwargs[key] = config.pop(key, value)

  # -----------------------
  # add env kwargs to path string. HACK
  # -----------------------
  # only use non-default arguments in path string
  env_path=dict()
  for k,v in env_kwargs.items():
    if v != default_env_kwargs[k]:
      env_path[k]=v
  if label:
    env_path['L']=label

  # -----------------------
  # get log dir for experiment
  # -----------------------
  log_path_config=dict(
    agent=agent,
    **env_path,
    **config
    )

  wandb_group = None
  if wandb_init_kwargs:
    wandb_group = wandb_init_kwargs.get('group', None)
  group = config.pop('group', wandb_group)

  # dir will be root_path/folder/group
  log_dir, exp_name = wandb_logger.gen_log_dir(
    base_dir=os.path.join(root_path, folder, group),
    hourminute=False,
    return_kwpath=True,
    date=False,
    path_skip=['num_steps', 'num_learner_steps', 'group'],
    **log_path_config
    )


  print("="*50)
  if os.path.exists(log_dir) and skip:
    print(f"SKIPPING\n{log_dir}")
    print("="*50)
    return
  else:
    print(f"RUNNING\n{log_dir}")
    print("="*50)

  # -----------------------
  # wandb settings
  # -----------------------
  if wandb_init_kwargs:
    wandb_init_kwargs['name']=exp_name # short display name for run
    if group is not None:
      wandb_init_kwargs['group']=group # short display name for run
    wandb_init_kwargs['dir'] = folder
  # needed for various services (wandb, etc.)
  os.chdir(root_path)

  # -----------------------
  # launch experiment
  # -----------------------

  if debug:
    config['num_steps'] = 50e3
  agent_config_file = f'{log_dir}/agent_config_kw.pkl'
  env_config_file = f'{log_dir}/env_config_kw.pkl'
  paths.process_path(log_dir)
  config_utils.save_config(agent_config_file, config)
  config_utils.save_config(env_config_file, env_kwargs)

  #TODO: could be made more general...
  command = make_program_command(
    agent=agent,
    wandb_init_kwargs=wandb_init_kwargs,
    folder=log_dir,
    agent_config=agent_config_file,
    env_config=env_config_file,
  )
  print(command)
  command = command.replace("\n", '')
  cuda_env = os.environ.copy()
  if cuda:
    cuda_env["CUDA_VISIBLE_DEVICES"] = str(cuda)
  process = subprocess.Popen(command, env=cuda_env, shell=True)
  process.wait()


def run(
    wandb_init_kwargs: dict,
    default_env_kwargs: dict,
    space: Union[Dict, List[Dict]],
    name: str ='train_many',
    use_wandb: bool = False,
    debug: bool = False,
    **kwargs):

  mp.set_start_method('spawn')
  root_path = str(Path().absolute())
  folder = FLAGS.folder
  skip = FLAGS.skip


  def train_function(config):
    """Run inside threads and creates new process.
    """
    p = mp.Process(
      target=create_and_run_program, 
      args=(config,),
      kwargs=dict(
        root_path=root_path,
        folder=folder,
        wandb_init_kwargs=wandb_init_kwargs if use_wandb else None,
        default_env_kwargs=default_env_kwargs,
        debug=debug,
        skip=skip,
        **kwargs)
      )
    p.start()
    wait_time = 30.0 # avoid collisions
    if wait_time and not debug:
      time.sleep(wait_time)
    p.join() # this blocks until the process terminates
    # this will call right away and end.

  if isinstance(space, dict):
    space = [space]

  from pprint import pprint
  pprint(space)

  experiment_specs = [tune.Experiment(
      name=name,
      run=train_function,
      config=s,
      resources_per_trial={"cpu": FLAGS.num_cpus, "gpu": FLAGS.num_gpus}, 
      local_dir='/tmp/ray',
    ) 
    for s in space
  ]
  tune.run_experiments(experiment_specs)

  import shutil
  if use_wandb:
    wandb_dir = wandb_init_kwargs.get("dir", './wandb')
    if os.path.exists(wandb_dir):
      shutil.rmtree(wandb_dir)
