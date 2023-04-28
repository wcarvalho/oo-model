import pickle
from absl import logging
from pprint import pprint
from ai2thor.controller import Controller
import random

def check_receptacle_clean(controller:Controller, target_obj,receptacle_type, random_seed):
    # sanitary check
    # clean receptacle if it was already contain the target_obj
    break_out_flag = True
    break_out_inner_loop = False
    while break_out_flag:
        for obj in controller.last_event.metadata["objects"]:
            break_out_inner_loop = False
            if obj["objectType"] == receptacle_type:
                child_id_list = obj["receptacleObjectIds"]
                # build a child list dict to look up target object
                if child_id_list is not None:
                    child_type_list = []
                    for i in range(len(child_id_list)):
                        tokens = child_id_list[i].split("|")
                        child_type_list.append(tokens[0])
                    idx_list = [range(1,len(child_type_list),1)]
                    child_dict = zip(child_type_list,idx_list)
                    # check if target object is in the child list of receptacle
                    if target_obj in child_dict :
                        break_out_inner_loop = True
                        break
        # random the current object again with excluded receptacle
        if break_out_inner_loop:
            random_seed = random.randrange(random_seed)
            controller.step(
                    action="InitialRandomSpawn",
                    randomSeed=random_seed,
                    forceVisible=False,
                    numPlacementAttempts=30,
                    placeStationary=True,
                    excludedReceptacles=[receptacle_type]
                )
            break_out_inner_loop = False
        else:
                break_out_flag = False


Navigation_action = [("MoveAhead"),("MoveBack"),("MoveLeft"),("MoveRight"),
                    ("LookUp"),("LookDown"),("RotateRight"),("RotateLeft")]
Interaction_action = [("PickupObject"), ("ReleaseObject")]
# totally 27 pairs
simple_basic_vector = [dict(x=0,y=0,z=0), dict(x=0.1,y=0,z=0),dict(x=0,y=0.1,z=0),dict(x=0,y=0,z=0.1),
                      dict(x=-0.1,y=0,z=0),dict(x=0,y=-0.1,z=0),dict(x=0,y=0,z=-0.1)]
basic_vector = [(0,0,0),
                (1,0,0),(0,1,0),(0,0,1),(-1,0,0),(0,-1,0),(0,0,-1),
                (1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0),
                (1,0,1),(-1,0,1),(1,0,-1),(-1,0,-1),
                (0,1,1),(0,-1,1),(0,1,-1),(0,-1,-1),
                (1,1,1),(-1,-1,-1),(-1,-1,1),(-1,1,1),
                (1,-1,-1),(1,1,-1),(1,-1,1),(-1,1,-1)
                ]
def load_config(filename):
  with open(filename, 'rb') as fp:
    config = pickle.load(fp)
    logging.info(f'Loaded: {filename}')
    return config


def save_config(filename, config):
  with open(filename, 'wb') as fp:
      def fits(x):
        y = isinstance(x, str)
        y = y or isinstance(x, float)
        y = y or isinstance(x, int)
        y = y or isinstance(x, bool)
        return y
      new = {k:v for k,v in config.items() if fits(v)}
      pickle.dump(new, fp)
      logging.info(f'Saved: {filename}')

def update_config(config, **kwargs):
  for k, v in kwargs.items():
    if not hasattr(config, k):
      raise RuntimeError(f"Attempting to set unknown attribute '{k}'")
    setattr(config, k, v)
