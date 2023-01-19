"""Class used for MultiLevel version of Kitchen Env.

Each level can have different distractors, different layout,
    different tasks, etc. Very flexible since just takes in 
    dict(level_name:level_kwargs).
"""
import numpy as np
import copy

from gym import spaces


from gym_minigrid.minigrid import Grid, WorldObj
from babyai.levels.levelgen import RoomGridLevel, RejectSampling


from envs.babyai_kitchen.world import Kitchen
import envs.babyai_kitchen.tasks
from envs.babyai_kitchen.levelgen import KitchenLevel

from envs.babyai.multilevel import MultiLevel as BabyAiMultiLevel

class MultiLevel(BabyAiMultiLevel):

  """main change is that 1 kitchen is created and copied to every environment. saves in loading time.
  Attributes:
      levelnames (list): names of levels
      levels (dict): level objects
  """

  def __init__(self,
      all_level_kwargs : dict,
      levelname2idx=None,
      LevelCls=KitchenLevel,
      wrappers=None,
      kitchen : Kitchen=None,
      path: str='.',
      **kwargs):
    """Summary
    
    Args:
        all_level_kwargs (dict): {levelname: kwargs} dictionary
        kitchen (Kitchen): Kitchen simulator to be used across envs.
        levelname2idx (dict, optional): {levelname: idx} dictionary. useful for returning idx versions of levelnames.
        **kwargs: kwargs for all levels
    """
    super().__init__(
      all_level_kwargs=all_level_kwargs,
      levelname2idx=levelname2idx,
      LevelCls=LevelCls,
      wrappers=wrappers,
      **kwargs)

    # -----------------------
    # initialize kitchen if not provided. 
    # use either kwargs or individual level
    #   kwargs to get settings
    # -----------------------
    if not kitchen:
      kitchen_kwargs = next(iter(self.all_level_kwargs.values()))
      if kwargs:
        kitchen_kwargs.update(kwargs)

      self.kitchen = Kitchen(
        objects=kitchen_kwargs.get('objects', []),
        tile_size=kitchen_kwargs.get('tile_size', 8),
        rootdir=kitchen_kwargs.get('root_dir', path),
        verbosity=kitchen_kwargs.get('verbosity', 0)
      )
  def create_level(self, **level_kwargs):
    level = self.LevelCls(
      # kitchen=copy.deepcopy(self.kitchen),
      kitchen=self.kitchen,
      **level_kwargs)
    return level
