from babyai.bot import Bot, GoNextToSubgoal
from gym_minigrid.minigrid import *
from babyai.levels.verifier import *
from babyai.levels.verifier import (ObjDesc, pos_next_to,
                                    GoToInstr, OpenInstr, PickupInstr, PutNextInstr, BeforeInstr, AndInstr, AfterInstr)
from typing import NamedTuple, List

class ActionsSubgoal(NamedTuple):
  """
  Attributes:
    memory: LSTM state
    sf: successor features
    policy_zeds: policy embeddings
  """
  goto: GoNextToSubgoal
  actions: List
