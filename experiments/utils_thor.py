# sanitary check
from ai2thor.controller import Controller
import random
from acme import wrappers
import env
from env import Ai2thorEnv
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import dm_env

def make_thor_environment(

    oar_wrapper: bool = True,
    ) -> dm_env.Environment:
  """Loads the Atari environment."""
  #keep_obj=["Apple", "Book"]
  keep_obj=None
  # switch to ManipulaTHOR
  controller = Controller(agentMode="arm",platform=CloudRendering)
  env = Ai2thorEnv(controller)

  # leave below alone.
  wrapper_list = [
      wrappers.SinglePrecisionWrapper,
  ]
  if oar_wrapper:
    # E.g. IMPALA and R2D2 use this particular variant.
    wrapper_list.append(wrappers.ObservationActionRewardWrapper)
  return wrappers.wrap_all(env, wrapper_list)

