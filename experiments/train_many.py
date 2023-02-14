from typing import Optional
from absl import flags
from absl import app
from absl import logging

import importlib
import functools
import multiprocessing as mp
import os
import time

from pathlib import Path
from ray import tune

from launchpad.nodes.python.local_multi_processing import PythonProcess
import launchpad as lp

from experiments import online_rl  # gets FLAGS arguments
from experiments import logger as wandb_logger 


flags.DEFINE_string('search', 'default', 'which search to use.')
flags.DEFINE_string('spaces', '', 'which search space to use.')

flags.DEFINE_integer('num_actors', 0, 'number of actors.')
flags.DEFINE_integer('num_cpus', 4, 'number of cpus.')
flags.DEFINE_integer('num_gpus', 1, 'number of gpus.')
flags.DEFINE_bool('skip', True, 'whether to skip things that have already run.')

FLAGS = flags.FLAGS

DEFAULT_NUM_ACTORS = 4
DEFAULT_LABEL = ''

def create_and_run_program(
    config,
    make_program_fn,
    root_path: str = '.',
    folder : Optional[str] = None,
    group : Optional[str] = None,
    wandb_init_kwargs: dict = None,
    default_env_kwargs: dict = None,
    terminal: str = 'current_terminal',
    skip: bool = True,
    debug: bool = False,
    log_every: float = 30.0):
  """Create and run launchpad program
  """

  # build_kwargs = build_kwargs or dict()
  agent = config.pop('agent', 'muzero')
  # num_actors = config.pop('num_actors', DEFAULT_NUM_ACTORS)
  cuda = config.pop('cuda', None)
  group = config.pop('group', group)
  label = config.pop('label', DEFAULT_LABEL)

  if cuda:
    os.environ['CUDA_VISIBLE_DEVICES']=str(cuda)

  # TODO: keep this variable?
  # save_config_dict=dict()

  # -----------------------
  # add env kwargs to path desc
  # -----------------------
  default_env_kwargs = default_env_kwargs or {}

  env_kwargs = dict()
  for key, value in default_env_kwargs.items():
    env_kwargs[key] = config.pop(key, value)

  # only use non-default
  env_path=dict()
  for k,v in env_kwargs.items():
    if v != default_env_kwargs[k]:
      env_path[k]=v
  if label:
    env_path['L']=label
    # save_config_dict['label'] = label

  # -----------------------
  # get log dir for experiment
  # -----------------------
  log_path_config=dict(
    agent=agent,
    **env_path,
    **config
    )
  log_dir, config_path_str = wandb_logger.gen_log_dir(
    base_dir=os.path.join(root_path, folder, group),
    hourminute=False,
    return_kwpath=True,
    date=False,
    path_skip=['max_number_of_steps'],
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
  name = config_path_str
  if wandb_init_kwargs:
    wandb_init_kwargs['name']=name # short display name for run
    if group is not None:
      wandb_init_kwargs['group']=group # short display name for run

  # needed for various services (wandb, etc.)
  os.chdir(root_path)

  # -----------------------
  # launch experiment
  # -----------------------
  program, local_resources = make_program_fn(config_kwargs=config,
                            env_kwargs=env_kwargs,
                            log_dir=log_dir,
                            path=root_path,
                            log_every=log_every,
                            wandb_init_kwargs=wandb_init_kwargs)

  # TODO: check that ray
  # if cuda:
  #   local_resources['learner'] = PythonProcess(
  #     env={"CUDA_VISIBLE_DEVICES": str(cuda)})

  if debug:
    print("="*50)
    print("LOCAL RESOURCES")
    print(local_resources)
    return

  controller = lp.launch(program,
    lp.LaunchType.LOCAL_MULTI_PROCESSING,
    terminal=terminal, 
    local_resources=local_resources
    )

  # -----------------------
  # blow is HUGE hack to cancel "cleanly"
  # get 1st process that exits to return to main program
  # use main program to send stop to all subprocesses
  # -----------------------
  controller.wait(return_on_first_completed=True)
  logging.warning("Search Process: Controller finished")
  time.sleep(60)
  controller._kill()


def main(_):
  mp.set_start_method('spawn')

  # -----------------------
  # wandb setup
  # -----------------------
  use_wandb = FLAGS.use_wandb
  search = FLAGS.search or 'default'
  group = FLAGS.wandb_group if FLAGS.wandb_group else search # overall group
  wandb_init_kwargs=dict(
    project=FLAGS.wandb_project,
    entity=FLAGS.wandb_entity,
    group=group, # overall group
    notes=FLAGS.wandb_notes,
    save_code=True,
  )
  # -----------------------
  # env setup
  # -----------------------
  folder = FLAGS.folder if FLAGS.folder else f"results/babyai"
  default_env_kwargs=dict(
    tasks_file='place_split_hard',
    )



  wait_time = 30.0
  terminal = 'current_terminal'
  num_cpus = FLAGS.num_cpus
  num_gpus = FLAGS.num_gpus
  skip = FLAGS.skip
  debug = FLAGS.debug
  space = importlib.import_module(f'{FLAGS.spaces}').get(FLAGS.search, FLAGS.agent)
  num_actors = FLAGS.num_actors or DEFAULT_NUM_ACTORS
  root_path = str(Path().absolute())

  def train_function(config):
    """Run inside threads and creates new process.
    """
    p = mp.Process(
      target=create_and_run_program, 
      args=(config,),
      kwargs=dict(
        make_program_fn=functools.partial(online_rl.make_distributed_program,
                                         num_actors=num_actors),
        root_path=root_path,
        folder=folder,
        group=group,
        wandb_init_kwargs=wandb_init_kwargs if use_wandb else None,
        default_env_kwargs=default_env_kwargs,
        terminal=terminal,
        debug=debug,
        skip=skip)
      )
    p.start()
    if wait_time and not debug:
      time.sleep(wait_time)
    p.join() # this blocks until the process terminates
    # this will call right away and end.

  experiment_specs = [tune.Experiment(
        name="goto",
        run=train_function,
        config=s,
        resources_per_trial={"cpu": num_cpus, "gpu": num_gpus}, 
        local_dir='/tmp/ray',
      ) 
      for s in space
  ]
  all_trials = tune.run_experiments(experiment_specs)

if __name__ == '__main__':
  app.run(main)
