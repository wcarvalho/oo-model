from typing import Optional, Union, List, Dict

from absl import flags, logging

import multiprocessing as mp
import os
import time
import pickle

import datetime
from pprint import pprint
from pathlib import Path
import subprocess

from acme.utils import paths

flags.DEFINE_integer('num_actors', 6, 'number of actors.')
flags.DEFINE_integer('config_idx', 1, 'number of actors.')
flags.DEFINE_integer('num_cpus', 16, 'number of cpus.')
flags.DEFINE_integer('num_gpus', 1, 'number of gpus.')
flags.DEFINE_integer('memory', 120_000, 'memory (in mbs).')
flags.DEFINE_string('time', '0-18:00:00', '18 hours.')
flags.DEFINE_integer('max_concurrent', 12, 'number of concurrent jobs')
flags.DEFINE_string('account', '', 'account on slurm servers to use.')
flags.DEFINE_string('partition', 'kempner', 'account on slurm servers to use.')

flags.DEFINE_string(
    'parallel', 'none', "none: run 1 experiment. sbatch: run many experiments with SBATCH. ray: run many experiments with say. use sbatch with SLUM or ray otherwise.")
flags.DEFINE_bool('subprocess', False, 'label for whether this run is a subprocess.')
flags.DEFINE_bool('debug_parallel', False, 'whether to debug parallel runs.')

flags.DEFINE_bool('skip', False, 'whether to skip experiments that have already run.')


FLAGS = flags.FLAGS

DEFAULT_LABEL = ''

def directory_not_empty(directory_path):
    return len(os.listdir(directory_path)) > 0

def date_time(time: bool=False):
  strkey = '%Y.%m.%d'
  if time:
    strkey += '-%H.%M'
  return datetime.datetime.now().strftime(strkey)

def gen_log_dir(
    base_dir="results/",
    date=False,
    hourminute=False,
    seed=None,
    return_kwpath=False,
    path_skip=[],
    **kwargs):

  kwpath = ','.join([f'{key[:4]}={value}' for key, value in kwargs.items() if not key in path_skip])

  if date:
    job_name = date_time(time=hourminute)
    path = Path(base_dir).joinpath(job_name).joinpath(kwpath)
  else:
    path = Path(base_dir).joinpath(kwpath)

  if seed is not None:
    path = path.joinpath(f'seed={seed}')

  if return_kwpath:
    return str(path), kwpath
  else:
    return str(path)

def get_all_configurations(spaces: Union[Dict, List[Dict]]):
    import itertools
    all_settings = []
    if isinstance(spaces, dict):
      spaces = [spaces]
    for space in spaces:
      # Extract keys and their corresponding lists from the space dictionary
      def get_values(d, key):
        vals = d[key]
        if isinstance(vals, dict) and 'grid_search' in vals:
          vals = vals['grid_search']
        elif isinstance(vals, (int, float, str)):
          vals = [vals]
        elif isinstance(vals, (list)):
          pass
        else:
          raise NotImplementedError(key, d)
        return key, vals

      keys, value_lists = zip(*[get_values(space, key) for key in space])

      # Generate the Cartesian product of the value lists
      cartesian_product = itertools.product(*value_lists)

      # Create a list of dictionaries for each combination
      all_settings += [dict(zip(keys, values)) for values in cartesian_product]

    return all_settings

def get_agent_env_configs(
    config: dict,
    neither: List[str] = ['group', 'label'],
    default_env_kwargs: Optional[dict]=None):
  """
  Separate config into agent and env configs. Example below. Basically if key starts with "env.", it goes into an env_config.
  Example:
  config = {
    seed: 1,
    width: 2,
    env.room_size: 7,
    group: 'wandb_group4'
  }
  agent_config = {seed: 1, width: 2}
  env_config = {room_size: 7}
  """
  agent_config = dict()
  env_config = dict()

  for k, v in config.items():
    if 'env.' in k:
      # e.g. "env.room_size"
      env_config[k.replace("env.", "")] = v
    elif default_env_kwargs and k in default_env_kwargs:
      # e.g. "room_size"
      env_config[k] = v
    elif k in neither:
      pass
    else:
      agent_config[k] = v
  
  return agent_config, env_config

def make_save_dict(
    config: dict,
    wandb_init_kwargs: dict,
    search_name: str,
    num_actors: int,
    run_distributed: bool,
    use_wandb: bool,
    base_dir: str = '.',
    ):
  # either look for group name in setting, wandb_init_kwargs, or use search name
  if 'group' in config:
    group = config.pop('group')
  else:
    group = wandb_init_kwargs.get('group', search_name)

  agent_config, env_config = get_agent_env_configs(
      config=config)

  # dir will be root_path/folder/group/exp_name
  # exp_name is also name in wandb
  log_dir, exp_name = gen_log_dir(
    base_dir=os.path.join(base_dir, group),
    return_kwpath=True,
    path_skip=['num_steps', 'num_learner_steps', 'group'],
    **agent_config,
    **env_config,
    )

  save_config = dict(
    agent_config=agent_config,
    env_config=env_config,
    use_wandb=use_wandb,
    wandb_group=group,
    wandb_name=exp_name,
    folder=log_dir,
    num_actors=num_actors,
    run_distributed=run_distributed,
    wandb_project=wandb_init_kwargs.get('project', None),
    wandb_entity=wandb_init_kwargs.get('entity', None),
  )

  return save_config

def make_program_command(
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
		--use_wandb=True
		--wandb_project='{wandb_project}'
		--wandb_entity='{wandb_entity}'
		--wandb_group='{wandb_group}'
    --wandb_name='{wandb_name}'
    --wandb_dir='{wandb_dir}'
    --folder='{folder}'
    --agent_config='{agent_config}'
    --env_config='{env_config}'
    --num_actors={num_actors}
    --run_distributed={run_distributed}
    --train_single=True
  """
  for k, v in kwargs.items():
    str += f"--{k}={v}"
  return str

def create_and_run_program(
    config,
    trainer_filename: str,
    root_path: str,
    folder: str,
    search_name: str,
    num_actors: int,
    run_distributed: bool,
    use_wandb: bool,
    wandb_init_kwargs: dict = None,
    skip: bool = True):
  """Create and run launchpad program
  """

  cuda = config.pop('cuda', None)

  # needed for various services (wandb, etc.)
  os.chdir(root_path)
  logging.info(f"Changed dir to {root_path}")
  base_dir = os.path.join(root_path, folder)

  #------------------------
  # save config
  #------------------------
  save_config = make_save_dict(
    config=config,
    wandb_init_kwargs=wandb_init_kwargs,
    search_name=search_name,
    num_actors=num_actors,
    run_distributed=run_distributed,
    use_wandb=use_wandb,
    base_dir=base_dir,
    )
  log_dir = save_config['folder']

  #------------------------
  # skip if already done
  #------------------------
  print("="*50)
  if skip and os.path.exists(log_dir) and directory_not_empty(log_dir):
    print(f"SKIPPING\n{log_dir}")
    print("="*50)
    return
  else:
    print(f"RUNNING\n{log_dir}")
    print("="*50)


  config_file = f'{log_dir}/config_kw.pkl'
  paths.process_path(log_dir)
  with open(config_file, 'wb') as fp:
      pickle.dump(save_config, fp)
      logging.info(f'Saved: {config_file}')

  # -----------------------
  # launch experiment
  # -----------------------
  command = f"python {trainer_filename}"
  command += f" --config_file='{config_file}'"
  command += f" --subprocess={True}"
  command += f" --make_path={False}"

  print(command)
  cuda_env = os.environ.copy()
  if cuda:
    cuda_env["CUDA_VISIBLE_DEVICES"] = str(cuda)
  process = subprocess.Popen(command, env=cuda_env, shell=True)
  process.wait()

def run_ray(
    trainer_filename: str,
    wandb_init_kwargs: dict,
    folder: str,
    search_name: str,
    spaces: Union[Dict, List[Dict]],
    use_wandb: bool = False,
    num_actors: int = 4,
    debug: bool = False,
    run_distributed: bool = True):

  skip = FLAGS.skip

  #--------------------------
  # setup base dir. important for ray because changes directory.
  #--------------------------
  root_path = str(Path().absolute())

  #--------------------------
  # setup spaces
  #--------------------------
  if isinstance(spaces, dict):
    spaces = [spaces]
  pprint(spaces)

  run_kwargs=dict(
    trainer_filename=trainer_filename,
    root_path=root_path,
    folder=folder,
    search_name=search_name,
    num_actors=num_actors,
    run_distributed=run_distributed,
    wandb_init_kwargs=wandb_init_kwargs,
    use_wandb=use_wandb,
    skip=skip)

  if debug:
    # just running first config
    configs = get_all_configurations(spaces=spaces)
    config = configs[0]
    config['num_steps'] = 50e3
    create_and_run_program(config, **run_kwargs)
    return

  #--------------------------
  # setup multi-processing + init ray
  #--------------------------
  from ray import tune
  mp.set_start_method('spawn')
  import ray
  ray.init()
  def train_function(config):
    """Run inside threads and creates new process.
    """
    p = mp.Process(
      target=create_and_run_program, 
      args=(config,),
      kwargs=run_kwargs
      )
    p.start()
    wait_time = 30.0 # avoid collisions
    if wait_time and not debug:
      time.sleep(wait_time)
    p.join() # this blocks until the process terminates
    # this will call right away and end.

  experiment_specs = [tune.Experiment(
      name=trainer_filename,
      run=train_function,
      config=s,
      resources_per_trial={"cpu": FLAGS.num_cpus, "gpu": FLAGS.num_gpus}, 
      local_dir='/tmp/ray',
    ) 
    for s in spaces
  ]
  tune.run_experiments(experiment_specs)

  import shutil
  if use_wandb:
    wandb_dir = wandb_init_kwargs.get("dir", './wandb')
    if os.path.exists(wandb_dir):
      shutil.rmtree(wandb_dir)


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

  import ray
  ray.init()


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

def run_sbatch(
    trainer_filename: str,
    wandb_init_kwargs: dict,
    folder: str,
    search_name: str,
    spaces: Union[Dict, List[Dict]],
    use_wandb: bool = False,
    num_actors: int = 4,
    debug: bool = False,
    run_distributed: bool = True):
  """For each possible configuration of a run, create a config entry. save a list of all config entries. When SBATCH is called, it will use the ${SLURM_ARRAY_TASK_ID} to run a particular one.
  """
  wandb_init_kwargs = wandb_init_kwargs or dict()
  #################################
  # create configs for all runs
  #################################
  root_path = str(Path().absolute())
  configurations = get_all_configurations(spaces=spaces)

  from pprint import pprint
  logging.info("searching:")
  pprint(configurations)


  save_configs = []
  base_path = os.path.join(root_path, folder, search_name)
  for config in configurations:
    save_config = make_save_dict(
      config=config,
      wandb_init_kwargs=wandb_init_kwargs,
      search_name=search_name,
      num_actors=num_actors,
      run_distributed=run_distributed,
      use_wandb=use_wandb,
      base_dir=base_path,
      )
    save_configs.append(save_config)

  #################################
  # save configs for all runs
  #################################
  base_path = os.path.join(base_path, f'runs-{date_time(True)}')
  paths.process_path(base_path)

  # base_filename = os.path.join(base_path, date_time(time=True))
  configs_file = f"{base_path}/config.pkl"
  with open(configs_file, 'wb') as fp:
      pickle.dump(save_configs, fp)
      logging.info(f'Saved: {configs_file}')

  #################################
  # create run.sh file to run with sbatch
  #################################
  python_file_contents = f"python {trainer_filename}"
  python_file_contents += f" --config_file={configs_file}"
  python_file_contents += f" --use_wandb={use_wandb}"
  python_file_contents += f" --num_actors={num_actors}"
  if debug:
    python_file_contents += f" --config_idx=1"
  else:
    python_file_contents += f" --config_idx=$SLURM_ARRAY_TASK_ID"
  python_file_contents += f" --run_distributed={run_distributed}"
  python_file_contents += f" --subprocess={True}"
  python_file_contents += f" --make_path={False}"

  run_file = f"{base_path}/run.sh"

  if debug:
    # create file and run single python command
    run_file_contents = "#!/bin/bash\n" + python_file_contents
    logging.warning("only running first config")
    print(run_file_contents)
    with open(run_file, 'w') as file:
      # Write the string to the file
      file.write(run_file_contents)
    process = subprocess.Popen(f"chmod +x {run_file}", shell=True)
    process = subprocess.Popen(run_file, shell=True)
    process.wait()
    return

  #################################
  # create sbatch file
  #################################
  job_name=f'{search_name}-{date_time(True)}'
  sbatch_contents = f"#SBATCH --gres=gpu:{FLAGS.num_gpus}\n"
  sbatch_contents += f"#SBATCH -c {FLAGS.num_cpus}\n"
  sbatch_contents += f"#SBATCH --mem {FLAGS.memory}\n"
  sbatch_contents += f"#SBATCH -J {job_name}\n"

  # sbatch_contents += f"#SBATCH --mem-per-cpu={FLAGS.memory}\n"
  sbatch_contents += f"#SBATCH -p {FLAGS.partition}\n"
  sbatch_contents += f"#SBATCH -t {FLAGS.time}"
  sbatch_contents += f"#SBATCH --account {FLAGS.account}\n"
  sbatch_contents += f"#SBATCH -o {base_path}/id=%j.out\n"
  sbatch_contents += f"#SBATCH -e {base_path}/id=%j.err\n"

  run_file_contents = "#!/bin/bash\n" + sbatch_contents + python_file_contents
  print("-"*20)
  print(run_file_contents)
  print("-"*20)
  with open(run_file, 'w') as file:
    # Write the string to the file
    file.write(run_file_contents)

  total_jobs = len(save_configs)
  max_concurrent = min(FLAGS.max_concurrent, total_jobs)
  sbatch_command = f"sbatch --array=1-{total_jobs}%{max_concurrent} {run_file}"
  logging.info(sbatch_command)
  process = subprocess.Popen(f"chmod +x {run_file}", shell=True)
  process = subprocess.Popen(sbatch_command, shell=True)
  process.wait()
