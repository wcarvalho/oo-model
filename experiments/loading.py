from typing import NamedTuple, Any
import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# https://github.com/google/jax/issues/8302
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import os.path
from acme import specs
from acme.jax import experiments
from acme.utils import counting
from acme.tf import savers

import jax
import numpy as np
from glob import glob

from experiments import config_utils as exp_utils
from experiments import babyai_factored_muzero

class LoadOutputs(NamedTuple):
  config: Any
  builder: Any
  learner: Any
  policy: Any
  checkpointer: Any
  actor: Any


def load_settings(
  base_dir: str = None,
  run: str = None,
  seed_path: str = None):
  # first load configs
  if seed_path is None:
    assert base_dir is not None and run is not None, 'set values for finding path'
    seed_path = glob(os.path.join(base_dir, run, '*'))[0]
  env_file = os.path.join(seed_path, 'env_config_kw.pkl')
  env_kwargs = exp_utils.load_config(env_file)

  config_file = os.path.join(seed_path, 'config.pkl')
  config_kwargs = exp_utils.load_config(config_file)
  return env_kwargs, config_kwargs


def load_agent(env,
               seed_path,
               config_kwargs,
               env_kwargs,
               use_latest = True,
               evaluation = True,
               agent_setup = babyai_factored_muzero.setup):
  config, builder, network_factory = agent_setup(
        launch=True,
        config_kwargs=config_kwargs,
        env_kwargs=env_kwargs)

  # then get environment spec
  environment_spec = specs.make_environment_spec(env)

  # the make network
  networks = network_factory(environment_spec)

  # make policy
  policy = builder.make_policy(
        networks=networks,
        environment_spec=environment_spec,
        evaluation=evaluation)

  # make learner
  key = jax.random.PRNGKey(config.seed)
  learner_key, key = jax.random.split(key)
  learner = builder.make_learner(
        random_key=learner_key,
        networks=networks,
        dataset=None,
        logger_fn=lambda x: None,
        environment_spec=environment_spec,
        replay_client=None,
        counter=None)

  # create checkpointer
  parent_counter = counting.Counter(time_delta=0.)

  # get all directories from year
  dirs = glob(os.path.join(seed_path, "*/checkpoints/learner")); 
  ckpts = glob(os.path.join(dirs[0], "*"))
  assert len(dirs) > 0

  checkpointing = experiments.CheckpointingConfig(
      directory=dirs[0],
      add_uid=False,
      max_to_keep=None,
  )

  checkpointer = savers.Checkpointer(
          objects_to_save={'learner': learner, 'counter': parent_counter},
          time_delta_minutes=checkpointing.time_delta_minutes,
          directory=checkpointing.directory,
          add_uid=checkpointing.add_uid,
          max_to_keep=checkpointing.max_to_keep,
          keep_checkpoint_every_n_hours=checkpointing.keep_checkpoint_every_n_hours,
          checkpoint_ttl_seconds=checkpointing.checkpoint_ttl_seconds,
      )

  reload(checkpointer, seed_path, use_latest)

  # make actor
  actor_key, key = jax.random.split(key)

  # will need a custom actor
  actor = builder.make_actor(
        actor_key, policy, environment_spec, variable_source=learner, adder=None)

  return LoadOutputs(
    config=config,
    builder=builder,
    learner=learner,
    policy=policy,
    actor=actor,
    checkpointer=checkpointer,
  )


def reload(checkpointer, seed_path, use_latest: bool = True):
  # get all directories from year
  dirs = glob(os.path.join(seed_path, "*/checkpoints/learner")); 
  ckpts = glob(os.path.join(dirs[0], "*"))
  # load checkpoint
  ckpts.sort()
  assert use_latest, 'need to implement otherwise'
  latest = ckpts[-1].split(".index")[0]
  ckpt_path = latest
  assert os.path.exists(f'{ckpt_path}.index')
  print('loading', ckpt_path)
  status = checkpointer._checkpoint.restore(ckpt_path)
