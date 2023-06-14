import numpy as np
from envs.babyai_kitchen.objects import KitchenObject, Food, KitchenContainer


class Kitchen(object):
  """docstring for Kitchen"""
  def __init__(self, objects=[], tile_size=32, rootdir='.', idx_offset=20, verbosity=0):
    super(Kitchen, self).__init__()

    self.carrying = None
    self.verbosity = verbosity

    self._objects = self._default_objects(rendering_scale=tile_size*3, rootdir=rootdir)

    # restrict objects
    if objects:
        self._objects = [o for o in self._objects if o.type in objects]

    # get # of states for objects
    max_obj_states = 0
    for idx, object in enumerate(self._objects):
      num_states = len(object.state2idx)
      max_obj_states = max(max_obj_states, num_states)

    self.objectid2object = {}
    for idx, object in enumerate(self._objects):
        start_idx = idx*max_obj_states
        object.set_verbosity(self.verbosity)
        object.set_id(start_idx + idx_offset)
        self.objectid2object[start_idx + idx_offset] = object


    self._max_states = object.object_id + max_obj_states
    self.reset()

    self._active = self._objects

  @property
  def max_object_state(self):
    return self._max_states
  

  def objects_with_property(self, props):
    return [object for object in self.objects 
        if sum([object.has_prop(p) for p in props]) == len(props)
    ]

  def objects_by_type(self, types, prop='type'):
    matches = []
    if isinstance(types, list):
        pass
    elif isinstance(types, str):
        types = [types]
    else:
        raise RuntimeError
    for t in types:
        matches.extend([object for object in self.objects if getattr(object, prop) == t])
    return matches

  # ======================================================
  # environment functions
  # ======================================================

  def update_carrying(self, carrying):
    self.carrying = carrying

  def set_active_objects(self, types):
    self._active = [o for o in self._objects if o.type in types]

  def reset(self, randomize_states=False):
    self.last_action_information = {}
    for object in self.objects:
        object.reset(random=randomize_states)
        assert object.contains is None
        assert object.init_pos is None

  def interact(self, action, object_infront, fwd_pos, grid, env):
    # Pick up an object

    if action in ('pickup_contents', 'pickup_container'):
        action_info = self.pickup_object(
            object_infront=object_infront,
            fwd_pos=fwd_pos,
            action_name=action,
            grid=grid)

    # place an object in front
    elif action == 'place':
        action_info = self.place(
            object_infront=object_infront,
            fwd_pos=fwd_pos,
            grid=grid)

    # Toggle/activate an object
    elif action == 'toggle':
        action_info = self.toggle(object_infront, env, fwd_pos)

    # slice
    elif action == 'slice':
        action_info = self.slice(object_infront)

    else:
        raise RuntimeError(f"Unknown action: {action}")

    return action_info

  def step(self):
    for object in self.objects:
        object.step()

  # ======================================================
  # Interactions
  # ======================================================

  def slice(self, object_infront):
    action_info = dict(
            action='slice',
            success=False,
            message='No object in front',
            )
    if object_infront and self.carrying:
        if hasattr(object_infront, 'kitchen_object'):
            action_info = object_infront.slice(self.carrying)
        else:
            action_info['message'] = f"{object_infront.type} isn't sliceable"

    return action_info

  def toggle(self, object_infront, env, fwd_pos):
    action_info = dict(
            action='toggle',
            success=False,
            message='No object in front',
            )
    if object_infront:
        if hasattr(object_infront, 'kitchen_object'):
            action_info = object_infront.toggle()
        else:
            # backwards compatibility
            action_info['success'] = object_infront.toggle(env, fwd_pos)

    return action_info

  def pickup_object(self, object_infront, fwd_pos, grid,
                    action_name: str, **kwargs):
    """pickup object. if container, pickup inner most object. 
    e.g., if applied to stove with pot with tomato. inner most is tomato.
    """
    action_info = dict(
        action=action_name,
        success=False,
        message='Nothing in front',
    )
    assert action_name in ('pickup_contents', 'pickup_container')
    if object_infront:
        if self.carrying is None:
            # ======================================================
            # pickup
            # ======================================================
            # for kitchen objects
            if hasattr(object_infront, 'kitchen_object'):
                if action_name == 'pickup_contents':
                  self.carrying, action_info = object_infront.pickup_contents()
                elif action_name == 'pickup_container':
                  self.carrying, action_info = object_infront.pickup_self()

                # -----------------------
                # update grid
                # -----------------------
                if self.carrying is not None:
                    if self.carrying.object_id == object_infront.object_id:
                        # set grid to None, ONLY if pickup container
                        self.carrying.cur_pos = np.array([-1, -1])
                        grid.set(*fwd_pos, None)

            # for older objects
            else:
                if object_infront.can_pickup():
                    self.carrying = object_infront
                    action_info['success'] = True
                    # update grid
                    if self.carrying is not None:
                        self.carrying.cur_pos = np.array([-1, -1])
                        grid.set(*fwd_pos, None)
                else:
                    action_info['message'] = f"Cannot pickup {object_infront.type}"

        else:
            action_info['message'] = f"could not {action_name} with {object_infront.type}. already carrying {self.carrying.type}"

    return action_info

  def place(self, object_infront, fwd_pos, grid):
    action_info = dict(
            action='place',
            success=False,
            message='Not carrying anything',
            )
    if self.carrying:

        # something in front
        if object_infront is not None:
          if hasattr(object_infront, "kitchen_object"):
            action_info = self.place_inside(object_infront)
          else:
            action_info['message'] = f"cannot place in/on {object_infront.type}"

        # nothing in front
        else:
            if not object_infront:
                action_info = self.carrying.place_on_tile(
                    grid=grid,
                    fwd_pos=fwd_pos)
                self.carrying = None
            else:
                action_info['message'] = f"cannot place in/on {object_infront.type}"

    return action_info

  def place_inside(self, container):
    # not container, keep object
    if not container.is_container:
       return dict(
            action='place_inside',
            success=False,
            message=f"can't place inside {container.type}. not a container",
            )

    # container is full
    if container.contains is not None:
        # try recursively placing. e.g. if stove, to put in pot
        return self.place_inside(container.contains)

    # container doesn't accept the type being carried
    if not container.accepts(self.carrying): 
       return dict(
            action='place_inside',
            success=False,
            message=f"can't place inside {container.type}. doesn't accept {self.carrying.type}",
            )

    action_info = container.place_inside(self.carrying)

    # no longer have object
    self.carrying = None
    return action_info

  # ======================================================
  # Objects
  # ======================================================
  @property
  def objects(self):
    return self._objects

  def _default_objects(self, rendering_scale=96, rootdir="."):
    foods = ['lettuce', 'potato', 'tomato', 'onion', 'apple', 'orange']
    nonfoods_pickupable = ['fork', 'knife', 'pot', 'pan', 'plates', 'bowl']
    pickupable = foods + nonfoods_pickupable
    return [
            KitchenContainer(
                name="sink",
                rendering_scale=rendering_scale,
                rootdir=rootdir,
                properties=['on', 'dirty'],
                visible_properties=['on'],
                can_contain=pickupable,
                pickupable=False,
                toggle_prop={'dirty': False},
            ),
            KitchenContainer(
                name="stove",
                rendering_scale=rendering_scale,
                rootdir=rootdir,
                properties=['on'],
                visible_properties=['on'],
                can_contain=['pot', 'pan'],
                pickupable=False,
                toggle_prop={'temp': 'hot'},
            ),
            KitchenContainer(
                name="microwave",
                rendering_scale=rendering_scale,
                rootdir=rootdir,
                properties=['on'],
                visible_properties=['on'],
                can_contain=['plates'],
                pickupable=False,
                toggle_prop={'temp': 'hot'},
            ),
            KitchenContainer(
                name="fridge",
                rendering_scale=rendering_scale,
                rootdir=rootdir,
                properties=['on'],
                visible_properties=[''],
                can_contain=pickupable,
                pickupable=False,
                toggle_prop={'temp': 'cold'},

            ),
            KitchenContainer(
                name="pot",
                rendering_scale=rendering_scale,
                rootdir=rootdir,
                # hides_content=True,
                can_contain=foods,
                properties=['dirty'],
                visible_properties=['dirty'],
                # can_heat_contained=True,
                # can_heat=True,
            ),
            KitchenContainer(
                name="pan",
                rendering_scale=rendering_scale,
                rootdir=rootdir,
                can_contain=foods,
                # hides_content=True,
                properties=['dirty'],
                visible_properties=['dirty'],
                # can_heat_contained=True,
                # can_heat=True,
            ),
            KitchenContainer(
                name="plates",
                rendering_scale=rendering_scale,
                rootdir=rootdir,
                can_contain=foods,
                properties=['dirty'],
                visible_properties=['dirty'],
            ),
            KitchenContainer(
                name="bowl",
                rendering_scale=rendering_scale,
                rootdir=rootdir,
                can_contain=foods,
                properties=['dirty'],
                visible_properties=['dirty'],
            ),

            KitchenObject(
                name="fork",
                object_type='utensil',
                rendering_scale=rendering_scale,
                rootdir=rootdir,
                properties=['dirty'],
                visible_properties=['dirty'],
            ),
            KitchenObject(
                name="knife",
                object_type='utensil',
                rendering_scale=rendering_scale,
                properties=['dirty'],
                visible_properties=['dirty'],
                rootdir=rootdir,
            ),
            Food(name='lettuce',
                rendering_scale=rendering_scale,
                rootdir=rootdir,
                ),
            Food(name='potato',
                rendering_scale=rendering_scale,
                rootdir=rootdir,
                ),
            Food(name='tomato',
                rendering_scale=rendering_scale,
                rootdir=rootdir,
                ),
            Food(name='onion',
                rendering_scale=rendering_scale,
                rootdir=rootdir,
                ),

            Food(name='apple',
                rendering_scale=rendering_scale,
                rootdir=rootdir,
                properties=['sliced'],
                visible_properties=['sliced'],
                ),

            Food(name='orange',
                rendering_scale=rendering_scale,
                rootdir=rootdir,
                properties=['sliced'],
                visible_properties=['sliced'],
                ),
    ]

if __name__ == '__main__':
    Food("knife")
