import os.path
import copy
from collections import namedtuple
from sklearn.model_selection import ParameterGrid
import numpy as np
from PIL import Image
from pprint import pprint

from gym_minigrid.minigrid import WorldObj
from gym_minigrid.rendering import fill_coords, point_in_rect


ICONPATH='envs/babyai_kitchen/icons'

def open_image(image, rendering_scale):
    if rendering_scale == 0: return None
    image = Image.open(image)
    arr = np.array(image.resize((rendering_scale, rendering_scale)))
    return arr

class KitchenObject(WorldObj):
  """docstring for KitchenObject"""
  def __init__(self,
        name,
        object_type='regular',
        pickupable=True,
        is_container=False,
        temp_decay=4,
        can_contain=[],
        rendering_scale=96,
        verbosity=0,
        rootdir='.',
        default_state=None,
        properties=[],
        visible_properties=[],
        property2states={'temp': ['cold', 'room', 'hot']},
        property2default={'temp': 'room'},
        toggle_prop={},
        ):
    """Load:
    - all possible object-states
    - image paths for object-states
    
    Args:
        name (TYPE): Description
        image_paths (None, optional): Description
        state (None, optional): Description
        state2idx (dict, optional): Description
        default_state_id (int, optional): Description
        rendering_scale (int, optional): Description
        verbosity (int, optional): Description
        properties (list, optional): Description
    """
    # super(KitchenObject, self).__init__()
    # ======================================================
    # load basics
    # ======================================================
    self.name = self.type = name
    self.object_type = object_type
    self.pickupable = pickupable

    self.temp_decay = temp_decay
    self.toggle_prop = toggle_prop

    self.is_container = is_container
    self.can_contain = can_contain
    self.verbosity = verbosity
    self.rendering_scale = rendering_scale
    self.properties = properties
    self.visible_properties = visible_properties if visible_properties else properties

    # add tempterature if not already there
    if not 'temp' in self.properties:
        self.properties.append('temp')

    # ======================================================
    # load possible object-states & images
    # ======================================================
    if properties:
        states = []
        state2idx = {}
        idx2state = {}
        image_paths = {}
        possible_states = {}
        for prop in properties:
            if prop in property2states:
                possible_states[prop]=property2states[prop]
            else:
                possible_states[prop]=[True, False]
        possible_states = [i for i in ParameterGrid(possible_states)]
        # -----------------------
        # load: paths, states
        # -----------------------

        for state in possible_states:
            # ensures that always matches ordering of list
            state = {p:state[p] for p in properties}
            key = str(state)
            states.append(key)

            # indx each state
            state2idx[key] = len(state2idx)
            idx2state[state2idx[key]] = state

            # get path for each state
            path = f"{name}"
            for prop in visible_properties:
                if prop and state[prop]:
                    path += f"_{prop}"
            image_paths[key] = os.path.join(rootdir, f"{ICONPATH}/{path}.png")

    else:
        image_paths = {'default':  f"{ICONPATH}/{name}.png"}
        state2idx = {'default':  0}
        states = ['default']
        idx2state = {0 : 'default'}


    self.image_paths = image_paths
    self.images = {k : open_image(v, rendering_scale) for k, v in image_paths.items()}

    # ======================================================
    # load state info
    # ======================================================

    self.idx2state = idx2state
    self.state2idx = state2idx
    self.states = states
    self.state = self.default_state = default_state
    if default_state:
      pass
    else:
        if properties:
            all_false = {prop: False for prop in properties}
            all_false.update(property2default)

            self.state = self.default_state = all_false
        else:
            self.state = self.default_state = "default"

    self.default_state_id = self.state2idx[str(self.default_state)]

    # ======================================================
    # reset position info
    # ======================================================
    self.reset(random=False)
    self.object_id = None
    self.kitchen_object = True

  def has_prop(self, prop):
    return prop in self.properties

  def set_prop(self, prop, val):
    self.state[prop] = val

  def render(self, screen):
    """Used for producing image
    """
    obj_img = self.state_image()
    np.copyto(dst=screen, src=obj_img[:, :, :3])
    fill_coords(screen, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(screen, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

  def reset(self, random=False):
    self.init_pos = None
    self.cur_pos = None
    self.contains = None
    self.reset_decay()

    if random:
        idx = np.random.randint(len(self.states))
    else:
        idx = self.default_state_id
    self.state = self.idx2state[idx]
    if self.verbosity > 1:
        print(f'{self.name} resetting to: {idx}/{len(self.states)} = {self.state}')

  def state_image(self):
    if self.verbosity > 1:
        print(f'object image: {self.name}, {self.state}')
    return self.images[str(self.state)]

  def set_id(self, oid): self.object_id = oid

  def state_id(self):
    return self.state2idx[str(self.state)]

  def encode(self):
    """Encode the a description of this object as a 3-tuple of integers"""
    raw_state_idx = self.state_id()
    object_wise_state_idx = self.object_id + raw_state_idx
    return (self.object_id, 0, object_wise_state_idx)

  @staticmethod
  def decode(type_idx, color_idx, state):
      import ipdb; ipdb.set_trace()

  def set_verbosity(self, v): self.verbosity = v

  def __repr__(self):
    state = self.state
    string = f"{self.name}: {str(state)}, since_decay: {self.steps_since_decay}, contains: ({str(self.contains)})"
    return string

  # ======================================================
  # Actions
  # ======================================================
  def action_info(self, name, success=True, message=''):
    info = dict(
        name=name,
        success=success,
        message=f'{self.name}: {message}',
    )

    return info

  def can_pickup(self): return self.pickupable

  def accepts(self, object):
      return object.type in self.can_contain

  def slice(self, carrying):
    can_slice = self.has_prop('sliced')
    # can't slice? action failed
    if not can_slice:
        return self.action_info(
            name='slice',
            success=False,
            message=f"not sliceable"
            )

    # already sliced, failed
    if self.state['sliced']:
        return self.action_info(
            name='slice',
            success=False,
            message=f"already sliced"
            )


    if carrying.type == 'knife':
        self.set_prop("sliced", True)
        # carrying.set_prop("dirty", True)
        return self.action_info(
            name='slice',
            )

    # wasn't carrying knife, failed
    return self.action_info(
        name='slice',
        success=True,
        message=f"need to carry knife. Was carrying {str(carrying.type)}."
        )


  def toggle(self):
    can_toggle = self.has_prop('on')
    # can't toggle, action fails
    if not can_toggle:
      return self.action_info(
        name='toggle',
        success=False,
        message='cannot toggle')

    # if on, toggle off
    if self.state['on']:
      self.set_prop("on", False)
      return self.action_info(
          name='toggle',
          success=True,
          message='turned off')

    # if off, toggle on
    else:
      self.set_prop("on", True)
      return self.action_info(
          name='toggle',
          success=True,
          message='turned on')


  def pickup_self(self):
    if self.can_pickup(): 
        return self, self.action_info(name='pickup_self')
    return None, self.action_info(
        name='pickup_self',
        success=False,
        message='cannot be picked up')

  def pickup_contents(self):
    """If has contents, return contents. Else, return self.
    """
    # 
    if self.contains is not None:
        contents, info = self.contains.pickup_contents()
        if contents == self.contains:
            """
            if tomato in pot, above will return the tomato.
            have to set contains to None.
            """
            self.contains = None
        else:
            """
            if tomato in pot in stove, above will return the tomato for the stove. contains remains in tact.
            """
            pass

        return contents, info
    else:
        return self.pickup_self()

  def apply_to_contents(self, change):
    """Only applies to children. not to self.
    """
    if self.contains:
        # first apply to contains
        for k, v in change.items():
            if self.contains.has_prop(k):
                self.contains.set_prop(k, v)

        # then contains will apply to what it contains
        self.contains.apply_to_contents(change)

  def reset_decay(self):
    self.steps_since_decay = 0
    if self.contains is not None:
        self.contains.reset_decay()

  def step(self):
    if self.has_prop('on') and self.state['on']:
        # set property from toggle (e.g. hot for stove)
        self.state.update(self.toggle_prop)

        # reset the decay for all children
        self.reset_decay()

        # is there something that is applied besides heating/cooling?
        if self.toggle_prop != 'temp':
            self.apply_to_contents(change=self.toggle_prop)


    if self.state['temp'] != 'room':
        # if not room temp, all children get this
        self.apply_to_contents(
            change={'temp': self.state['temp']}
            )
        if self.contains:
            self.contains.reset_decay()

    else:
        # if room temp, no need for decay
        self.steps_since_decay = 0

    # after some number of steps, set temp to room
    if self.steps_since_decay >= self.temp_decay:
        self.state['temp'] = 'room'

    self.steps_since_decay += 1



class Food(KitchenObject):
  """docstring for Food"""
  def __init__(self,
    name,
    properties=['sliced', 'cooked'],
    visible_properties=['sliced', 'cooked'],
    **kwargs):
    super(Food, self).__init__(
        name=name,
        properties=properties,
        visible_properties=visible_properties,
        object_type='food',
         **kwargs)

  def step(self):
    super(Food, self).step()
    # food get's cooked when hot
    if self.has_prop('cooked') and self.state['temp'] == 'hot':
        self.set_prop("cooked", True)


class KitchenContainer(KitchenObject):
  """docstring for KitchenContainer"""
  def __init__(self, *args, **kwargs):
    super(KitchenContainer, self).__init__(*args, 
        is_container=True,
        object_type='container',
        **kwargs)

    assert self.can_contain, "must accept things"

