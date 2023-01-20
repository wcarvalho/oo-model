import sys
import re
import numpy as np
import functools
from babyai.levels.verifier import Instr
from envs.babyai_kitchen.world import Kitchen
from envs.babyai_kitchen.types import ActionsSubgoal

from babyai.bot import Bot, GoNextToSubgoal
from babyai.levels.verifier import (ObjDesc, pos_next_to,
                            GoToInstr, OpenInstr, PickupInstr, PutNextInstr, BeforeInstr, AndInstr, AfterInstr)

def reject_next_to(env, pos):
  """
  Function to filter out object positions that are right next to
  the agent's starting point
  """

  sx, sy = env.agent_pos
  x, y = pos
  d = abs(sx - x) + abs(sy - y)
  return d < 2

def get_matching_objects(kitchen, object_types=None, matchfn=None):
  """Get objects matching conditions
  
  Args:
      kitchen (Kitchen): kitchen world containing objects
      object_types (None, optional): list of object types to sample from
      matchfn (TYPE, optional): criteria on objects to use for selecting
          options
  
  Returns:
      list: objects
  """
  if object_types is None and matchfn is None:
      return []

  if object_types:
      return kitchen.objects_by_type(object_types)
  else:
      return [o for o in kitchen.objects if matchfn(o)]

def pickedup(kitchen, obj):
  return kitchen.carrying.type == obj.type

def remove_excluded(objects, exclude):
  return [o for o in objects if not o.type in exclude]

class KitchenTask(Instr):
  """docstring for KitchenTasks"""
  def __init__(self,
    env,
    kitchen,
    done_delay=0,
    reward=1.0,
    verbosity=0.0,
    argument_options=None,
    task_reps=None,
    use_subtasks=False,
    negate=False,
    reset_behavior: str='none',
    init=True):
    """Summary
    
    Args:
        env (LevelGen): Description
        kitchen (Kitchen): Description
        done_delay (int, optional): time-steps before returning done after complete
        reward (float, optional): reward at end
        argument_options (None, optional): object types to be sampled from
        task_reps (None, optional): representation of tasks
        use_subtasks (bool, optional): use subtasks
        negate (bool, optional): reward is negated and "not" is added before task description
        reset_behavior (str, optional): 'none': nothing happens.
          'remove': task objects are removed.
          'respawn': task objects are respawned.
        init (bool, optional): initialize task
    """
    super(KitchenTask, self).__init__()
    self.argument_options = argument_options or dict(x=[])
    self._task_objects = []
    self.negate = negate
    self.verbosity = verbosity
    self.env = env
    self._reward = reward
    if negate:
      self._reward *= -1
    self.kitchen = kitchen
    self.use_subtasks = use_subtasks
    self.finished = False
    self.done = False
    self._task_reps = task_reps
    self.reset_behavior = reset_behavior
    self.done_delay = done_delay
    self._time_complete = 0
    self._task_name = 'kitchentask'
    if init:
      self.instruction = self.generate()
    else:
      self.instruction = ''

    # check that object-states are valid by querying state id
    for object in self.task_objects:
      object.state_id()

  # -----------------------
  # generating task
  # -----------------------
  def generate(self, exclude=[], argops=None):
    raise NotImplemented

  # -----------------------
  # resetting task
  # -----------------------
  def reset(self, exclude=[]):
    self.instruction = self.generate(exclude)

  def reset_task(self):
    self.reset_objects()
    self.finished = False
    self._time_complete = 0

  def reset_objects(self):
    raise NotImplemented

  # -----------------------
  # checking task status
  # -----------------------
  def check_status(self):
    """Check is goal state has been achieved
    
    Returns:
        TYPE: Description
    """
    return False, False

  def update_status(self, goals_achieved: bool=False):
    """If goals_achieved, set task to "finished". Return done after the 
    required number of delay steps before returning done.
    
    Args:
        goals_achieved (bool, optional): Description
    
    Returns:
        TYPE: Description
    """
    reward = 0.0
    done = False
    if self.finished:
      self._time_complete += 1
    else:
      if goals_achieved:
        self.finished = True
        reward = self._reward

    if self._time_complete >= self.done_delay:
      done = True

    return reward, done

  def check_and_update_status(self):
    """Summary
    """
    goals_achieved, task_done = self.check_status()
    reward, done = self.update_status(goals_achieved)

    return reward, done

  @property
  def default_task_rep(self):
    raise NotImplemented

  @property
  def task_rep(self):
    if self._task_reps is not None:
      rep = self._task_reps.get(self.task_name, self.default_task_rep)
    else:
      rep = self.default_task_rep

    if self.negate:
      rep = "not " + rep
    return rep

  @property
  def task_name(self):
    return self._task_name

  @property
  def task_objects(self):
    return self._task_objects

  @property
  def task_types(self):
    return [o.type for o in self._task_objects]

  def surface(self, *args, **kwargs):
    return self.instruction

  @property
  def num_navs(self): return 1

  def __repr__(self):
    string = self.instruction
    if self.task_objects:
        for object in self.task_objects:
            string += "\n" + str(object)

    return string

  def check_actions(self, actions):
    for action in self.task_actions():
        if action == 'pickup':
            assert 'pickup_contents' in actions or 'pickup_container' in actions
        elif action == 'pickup_and':
            assert 'pickup_contents' in actions and 'pickup_container' in actions
        else:
            assert action in actions

  @staticmethod
  def task_actions():
    return [
        'toggle',
        'pickup_and',
        'place'
        ]

# ======================================================
# Base Tasks
# ======================================================
# ---------------
# Length = 1
# ---------------
class PickupTask(KitchenTask):

  @property
  def default_task_rep(self): return 'pickup x'

  @property
  def task_name(self):
    return 'pickup'

  def generate(self, exclude=[], argops=None):
    # which option to pickup
    pickup_objects = get_matching_objects(self.kitchen,
        object_types=argops or self.argument_options.get('x', []),
        matchfn=lambda o:o.pickupable)
    pickup_objects = remove_excluded(pickup_objects, exclude)
    self.pickup_object = np.random.choice(pickup_objects)

    self._task_objects = [
        self.pickup_object, 
    ]

    return self.task_rep.replace('x', self.pickup_object.type)

  def reset_objects(self):
    pass

  def check_status(self):
    if self.kitchen.carrying:
        done = reward = self.kitchen.carrying.type == self.pickup_object.type
    else:
        done = reward = False

    return reward, done

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.pickup_object, actions=['pickup_contents'])
    ]

class ToggleTask(KitchenTask):
  @property
  def default_task_rep(self): return 'turnon x'

  @property
  def task_name(self): return 'toggle'

  def reset_objects(self):
    self.toggle.set_prop("on", False)

  def generate(self, exclude=[], argops=None):

    x_options = argops or self.argument_options.get('x', [])
    if x_options:
        totoggle_options = self.kitchen.objects_by_type(x_options)
    else:
        totoggle_options = self.kitchen.objects_with_property(['on'])

    totoggle_options = remove_excluded(totoggle_options, exclude)
    self.toggle = np.random.choice(totoggle_options)

    self.toggle.set_prop("on", False)

    self._task_objects = [
        self.toggle,
    ]
    instr = self.task_rep.replace(
      'x', self.toggle.name)

    return instr

  @property
  def num_navs(self): return 2

  def check_status(self):
    reward = done = self.toggle.state['on'] == True
    return reward, done

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.toggle, actions=['toggle']),
    ]

# ---------------
# Length = 2
# ---------------

class HeatTask(KitchenTask):
  @property
  def default_task_rep(self): return 'heat x'

  @property
  def task_name(self): return 'heat'

  def generate(self, exclude=[], argops=None):
    self.stove = self.kitchen.objects_by_type(['stove'])[0]

    x_options = argops or self.argument_options.get('x', [])
    if x_options:
        objects_to_heat = self.kitchen.objects_by_type(x_options)
    else:
        objects_to_heat = self.kitchen.objects_by_type(self.stove.can_contain)
    objects_to_heat = remove_excluded(objects_to_heat, exclude)
    self.object_to_heat = np.random.choice(objects_to_heat)


    self.object_to_heat.set_prop("temp", "room")
    self.stove.set_prop("temp", 'room')
    self.stove.set_prop("on", False)


    self._task_objects = [
        self.object_to_heat,
        self.stove,
    ]
    return self.task_rep.replace('x', self.object_to_heat.name)

  @property
  def num_navs(self): return 2

  def check_status(self):
    done = reward = self.object_to_heat.state['temp'] == 'hot'
    return reward, done

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.object_to_heat, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.stove, actions=['place', 'toggle'])
    ]

class CleanTask(KitchenTask):

  @property
  def task_name(self): return 'clean'

  @property
  def default_task_rep(self): return 'clean x'

  def generate(self, exclude=[], argops=None):
    x_options = argops or self.argument_options.get('x', [])
    exclude = ['sink']+exclude
    if x_options:
        objects_to_clean = self.kitchen.objects_by_type(x_options)
    else:
        objects_to_clean = self.kitchen.objects_with_property(['dirty'])

    objects_to_clean = remove_excluded(objects_to_clean, exclude)
    self.object_to_clean = np.random.choice(objects_to_clean)
    self.object_to_clean.set_prop('dirty', True)


    self.sink = self.kitchen.objects_by_type(["sink"])[0]
    self.sink.set_prop('on', False)

    self._task_objects = [self.object_to_clean, self.sink]

    return self.task_rep.replace('x', self.object_to_clean.name)

  def reset_objects(self):
    self.object_to_clean.set_prop('dirty', True)
    self.sink.set_prop('on', False)

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.object_to_clean, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.sink, actions=['place', 'toggle'])
    ]

  @property
  def num_navs(self): return 2

  def check_status(self):
    done = reward = self.object_to_clean.state['dirty'] == False

    return reward, done

class SliceTask(KitchenTask):
  """docstring for SliceTask"""

  @property
  def task_name(self): return 'slice'

  @property
  def default_task_rep(self): return 'slice x'

  def get_options(self, exclude=[], argops=None):
    x_options = argops or self.argument_options.get('x', [])
    if x_options:
        objects_to_slice = self.kitchen.objects_by_type(x_options)
    else:
        objects_to_slice = self.kitchen.objects_with_property(['sliced'])
    objects_to_slice = remove_excluded(objects_to_slice, exclude)
    return {'x': objects_to_slice}

  def reset_objects(self):
    self.object_to_slice.set_prop('sliced', False)
    self.knife.set_prop('dirty', False)

  def generate(self, exclude=[], argops=None):

    objects_to_slice = self.get_options(exclude, argops)['x']
    self.object_to_slice = np.random.choice(objects_to_slice)
    self.object_to_slice.set_prop('sliced', False)

    self.knife = self.kitchen.objects_by_type(["knife"])[0]

    self._task_objects = [self.object_to_slice, self.knife]
    return self.task_rep.replace('x', self.object_to_slice.name)

  @property
  def num_navs(self): return 2

  def check_status(self):
    done = reward = self.object_to_slice.state['sliced'] == True

    return reward, done

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.knife, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.object_to_slice, actions=['slice'])
    ]

  @staticmethod
  def task_actions():
    return [
        'slice',
        'pickup_and',
        'place'
        ]

class ChillTask(KitchenTask):
  """docstring for CookTask"""

  @property
  def task_name(self): return 'chill'

  @property
  def default_task_rep(self): return 'chill x'

  def generate(self, exclude=[], argops=None):
    self.fridge = argops or self.kitchen.objects_by_type(['fridge'])[0]

    x_options = self.argument_options.get('x', [])
    if x_options:
        objects_to_chill = self.kitchen.objects_by_type(x_options)
    else:
        objects_to_chill = self.kitchen.objects_by_type(self.fridge.can_contain)

    objects_to_chill = remove_excluded(objects_to_chill, exclude)
    self.object_to_chill = np.random.choice(objects_to_chill)


    self.object_to_chill.set_prop("temp", "room")
    self.fridge.set_prop("temp", 'room')
    self.fridge.set_prop("on", False)


    self._task_objects = [
        self.object_to_chill,
        self.fridge,
    ]
    return self.task_rep.replace('x', self.object_to_chill.name)

  def reset_objects(self):
    self.object_to_chill.set_prop("temp", "room")
    self.fridge.set_prop("temp", 'room')
    self.fridge.set_prop("on", False)

  @property
  def num_navs(self): return 2

  def check_status(self):
    done = reward = self.object_to_chill.state['temp'] == 'cold'

    return reward, done

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.object_to_chill, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.fridge, actions=['place', 'toggle'])
    ]

class PickupCleanedTask(CleanTask):
  """docstring for CleanTask"""

  @property
  def task_name(self): return 'pickup_cleaned'

  @property
  def default_task_rep(self): return 'cleaned x'

  def check_status(self):
    if self.kitchen.carrying:
        clean = self.object_to_clean.state['dirty'] == False
        picked_up = self.kitchen.carrying.type == self.object_to_clean.type
        reward = done = clean and picked_up
    else:
        done = reward = False

    return reward, done

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.object_to_clean, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.sink, actions=['place', 'toggle', 'pickup_contents'])
    ]

class SlicePutdownTask(SliceTask):
  """docstring for SliceTask"""

  @property
  def task_name(self): return 'slice_putdown'
  @property
  def default_task_rep(self): return 'slice x and drop knife'

  @property
  def num_navs(self): return 2

  def check_status(self):
    sliced = self.object_to_slice.state['sliced'] == True
    not_carrying_knife = self.kitchen.carrying is not self.knife
    reward = done = sliced and not_carrying_knife

    return reward, done


  def subgoals(self):
    # TODO: rotate and try to place is a hack that should be replaced
    return [
      ActionsSubgoal(
        goto=self.knife, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.object_to_slice, actions=['slice', *(['left', 'place']*4)])
    ]

class PickupSlicedTask(SliceTask):
  """docstring for SliceTask"""

  @property
  def task_name(self): return 'pickup_sliced'
  @property
  def default_task_rep(self): return 'sliced x'

  @property
  def num_navs(self): return 2

  def check_status(self):
    if self.kitchen.carrying:
        sliced = self.object_to_slice.state['sliced'] == True
        picked_up = self.kitchen.carrying.type == self.object_to_slice.type
        reward = done = sliced and picked_up
    else:
        done = reward = False

    return reward, done


  def subgoals(self):
    # TODO: rotate and try to place is a hack that should be replaced
    return [
      ActionsSubgoal(
        goto=self.knife, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.object_to_slice, actions=['slice', *(['left', 'place']*4), 'pickup_contents'])
    ]

class PickupChilledTask(ChillTask):
  """docstring for CookTask"""

  @property
  def task_name(self): return 'pickup_chilled'
  @property
  def default_task_rep(self): return 'pickup chilled x'

  @property
  def num_navs(self): return 2

  def check_status(self):
      if self.kitchen.carrying:
          chilled = self.object_to_chill.state['temp'] == 'cold'
          picked_up = self.kitchen.carrying.type == self.object_to_chill.type
          reward = done = chilled and picked_up
      else:
          done = reward = False

      return reward, done

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.object_to_chill, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.fridge, actions=['place', 'toggle', 'pickup_contents'])
    ]

class PickupHeatedTask(HeatTask):
  """docstring for CookTask"""

  @property
  def task_name(self): return 'pickup_heated'
  @property
  def default_task_rep(self): return 'heated x'

  @property
  def num_navs(self): return 2

  def check_status(self):
    if self.kitchen.carrying:
        heated = self.object_to_heat.state['temp'] == 'hot'
        picked_up = self.kitchen.carrying.type == self.object_to_heat.type
        reward = done = heated and picked_up
    else:
        done = reward = False

    return reward, done

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.object_to_heat, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.stove, actions=['place', 'toggle', 'pickup_contents'])
    ]

class PlaceTask(KitchenTask):

  @property
  def task_name(self): return 'place'
  @property
  def default_task_rep(self): return 'place x on y'

  def generate(self, exclude=[], argops=None):
    if exclude:
      raise NotImplementedError
    if argops:
      raise NotImplementedError
    # -----------------------
    # get possible containers/pickupable objects
    # -----------------------
    x_options = self.argument_options.get('x', [])
    y_options = self.argument_options.get('y', [])

    pickup_type_objs = get_matching_objects(self.kitchen,
        object_types=x_options,
        matchfn=lambda o:o.pickupable)
    container_type_objs = get_matching_objects(self.kitchen,
        object_types=y_options,
        matchfn=lambda o:o.is_container)

    if y_options and x_options:
        # pick container
        self.container = np.random.choice(container_type_objs)

        # pick objects which can be recieved by container
        pickup_types = [o.type for o in pickup_type_objs]
        pickup_types = [o for o in self.container.can_contain
                            if o in pickup_types]
        pickup_type_objs = self.kitchen.objects_by_type(pickup_types)
        assert len(pickup_type_objs) > 0, "no match found"

        # pick 1 at random
        self.to_place = np.random.choice(pickup_type_objs)

    elif x_options:
        # sample pickup first
        self.to_place = np.random.choice(pickup_type_objs)

        # restrict to wich can accept to_place
        container_type_objs = [o for o in self.kitchen.objects 
                                if o.accepts(self.to_place)]
        assert len(container_type_objs) > 0, "no match found"

        # pick 1 at random
        self.container = np.random.choice(container_type_objs)
    else:
        # pick container
        self.container = np.random.choice(container_type_objs)
        # pick thing that can be placed inside
        pickup_type_objs = self.kitchen.objects_by_type(self.container.can_contain)
        self.to_place = np.random.choice(pickup_type_objs)


    self._task_objects = [
        self.container, 
        self.to_place
    ]

    task = self.task_rep.replace('x', self.to_place.name)
    task = task.replace('y', self.container.name)
    return task

  def check_status(self):
    if self.container.contains:
        # let's any match fit, not just the example used for defining the task. 
        # e.g., if multiple pots, any pot will work inside container
        done = reward = self.container.contains.type == self.to_place.type
    else:
        done = reward = False

    return reward, done

  @staticmethod
  def task_actions():
    return [
        'pickup_and',
        'place'
        ]

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.to_place, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.container, actions=['place'])
    ]

class PickupPlacedTask(KitchenTask):

  @property
  def task_name(self): return 'pickup_placed'
  @property
  def default_task_rep(self): return 'pickup x on y'

  def check_status(self):
    placed, placed = super().__check_status(self)
    carrying = pickedup(self.kitchen, self.container)
    done = reward = carrying and placed
    return reward, done

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.to_place, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.container, actions=['place', 'pickup_container'])
    ]

# ---------------
# Length 3
# ---------------

class CookTask(KitchenTask):
  """docstring for CookTask"""

  @property
  def task_name(self): return 'cook'
  @property
  def default_task_rep(self): return 'cook x with y on z'

  def generate(self, exclude=[], argops=None):
    if argops:
      raise NotImplementedError
    else:
      x_options = self.argument_options.get('x', [])
      y_options = self.argument_options.get('y', [])
      if self.argument_options.get('z', None) is not None:
        raise NotImplementedError 

    if x_options:
        objects_to_cook = self.kitchen.objects_by_type(x_options)
    else:
        objects_to_cook = self.kitchen.objects_with_property(['cooked']) # x

    if y_options:
        objects_to_cook_with = self.kitchen.objects_by_type(y_options)
    else:
        objects_to_cook_with = self.kitchen.objects_by_type(['pot', 'pan']) # y

    objects_to_cook_with = remove_excluded(objects_to_cook_with, exclude)
    objects_to_cook = remove_excluded(objects_to_cook, exclude)
    self.object_to_cook_on = self.kitchen.objects_by_type(['stove'])[0]
    self.object_to_cook = np.random.choice(objects_to_cook)
    self.object_to_cook_with = np.random.choice(objects_to_cook_with)

    self.object_to_cook.set_prop("cooked", False)
    self.object_to_cook.set_prop("temp", 'room')
    self.object_to_cook_with.set_prop("dirty", False)
    self.object_to_cook_on.set_prop("on", False)


    self._task_objects = [
        self.object_to_cook,
        self.object_to_cook_with,
        self.object_to_cook_on
    ]

    task = self.task_rep.replace(
      'x', self.object_to_cook.name).replace(
      'y', self.object_to_cook_with.name).replace(
      'z', self.object_to_cook_on.name)
    return task

  @property
  def num_navs(self): return 3

  def check_status(self):
    done = reward = self.object_to_cook.state['cooked'] == True

    return reward, done

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.object_to_cook, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.object_to_cook_with, actions=['place', 'pickup_container']),
      ActionsSubgoal(
        goto=self.object_to_cook_on, actions=['place', 'toggle'])
    ]

class PickupCookedTask(CookTask):

  @property
  def task_name(self): return 'pickup_cooked'
  @property
  def default_task_rep(self): return 'pickup cooked x'


  @property
  def num_navs(self): return 3

  def check_status(self):
      _, done = super().check_status()
      if self.kitchen.carrying:
          picked_up = self.kitchen.carrying.type == self.object_to_cook.type
          reward = done = done and picked_up
      else:
          done = reward = False

      return reward, done


  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.object_to_cook, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.object_to_cook_with, actions=['place', 'pickup_container']),
      ActionsSubgoal(
        goto=self.object_to_cook_on, actions=['place', 'toggle', 'pickup_contents'])
    ]

class PlaceSlicedTask(SliceTask):
  """docstring for SliceTask"""

  @property
  def task_name(self): return 'place_sliced'
  @property
  def default_task_rep(self): return 'place sliced x on y'

  def generate(self, exclude=[], argops=None):
    if exclude or argops:
      raise NotImplementedError

    # -----------------------
    # x= object to slice
    # -----------------------
    x_options = self.argument_options.get('x', [])
    if x_options:
        objects_to_slice = self.kitchen.objects_by_type(x_options)
    else:
        objects_to_slice = self.kitchen.objects_with_property(['sliced'])
    objects_to_slice = remove_excluded(objects_to_slice, exclude)
    self.object_to_slice = np.random.choice(objects_to_slice)
    self.object_to_slice.set_prop('sliced', False)

    # -----------------------
    # knife
    # -----------------------
    self.knife = self.kitchen.objects_by_type(["knife"])[0]


    # -----------------------
    # y = container
    # -----------------------

    # restrict to wich can accept to_place
    container_type_objs = [o for o in self.kitchen.objects 
                            if o.accepts(self.object_to_slice)]
    container_type_objs = remove_excluded(container_type_objs, exclude)
    assert len(container_type_objs) > 0, "no match found"

    # pick 1 at random
    self.container = np.random.choice(container_type_objs)

    self._task_objects = [self.object_to_slice, self.knife, self.container]
    return self.task_rep.replace(
      'x', self.object_to_slice.name).replace(
      'y', self.container.name)

  @property
  def num_navs(self): return 2

  def check_status(self):
    if self.container.contains:
        # let's any match fit, not just the example used for defining the task. 
        # e.g., if multiple pots, any pot will work inside container
        object_sliced = self.object_to_slice.state['sliced'] == True
        placed = self.container.contains.type == self.object_to_slice.type
        done = reward = object_sliced and placed
    else:
        object_sliced = False
        done = reward = False
    if self.verbosity:
      print(f"sliced={object_sliced}, placed={placed}")
    return reward, done

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.knife, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.object_to_slice, actions=['slice', *(['left', 'place']*4), 'pickup_contents']),
      ActionsSubgoal(
        goto=self.container, actions=['place']),
    ]

class PlaceCleanedTask(CleanTask):
  """docstring for PlaceCleanedTask"""

  @property
  def task_name(self): return 'place_cleaned'
  @property
  def default_task_rep(self): return 'place cleaned x on y'

  def generate(self, exclude, argops=None):
    super(PlaceCleanedTask, self).generate(exclude+['bowl', 'plates'], argops)

    # -----------------------
    # y = container
    # -----------------------
    # restrict to wich can accept to_place
    container_type_objs = [o for o in self.kitchen.objects 
                            if o.accepts(self.object_to_clean)]

    container_type_objs = remove_excluded(container_type_objs, exclude+['sink'])
    assert len(container_type_objs) > 0, f"no match found {self.object_to_clean.name}"

    # pick 1 at random
    self.container = np.random.choice(container_type_objs)

    self._task_objects += [self.container]

    return self.task_rep.replace(
      'x', self.object_to_clean.name).replace(
      'y', self.container.name)

  def check_status(self):
    _, cleaned = super(PlaceCleanedTask, self).check_status()
    if self.container.contains:
        # let's any match fit, not just the example used for defining the task. 
        # e.g., if multiple pots, any pot will work inside container
        placed = self.container.contains.type == self.object_to_clean.type
        done = reward = cleaned and placed
    else:
        placed = False
        done = reward = False
    if self.verbosity:
      print(f"cleaned={cleaned}, placed={placed}")
    return reward, done

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.object_to_clean, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.sink, actions=['place', 'toggle', 'pickup_contents']),
      ActionsSubgoal(
        goto=self.container, actions=['place'])
      ]

# -----------------------
# Length 4
# -----------------------
class PlaceCookedTask(CookTask):
  """docstring for PlaceCookedTask"""

  @property
  def task_name(self): return 'place_cooked'
  @property
  def default_task_rep(self): return 'place cooked x on y'

  def generate(self, exclude, argops=None):
    super(PlaceCookedTask, self).generate(exclude, argops)

    # -----------------------
    # y = container
    # -----------------------
    # restrict to wich can accept to_place
    container_type_objs = [o for o in self.kitchen.objects 
                            if o.accepts(self.object_to_cook)]

    container_type_objs = remove_excluded(container_type_objs, exclude)
    assert len(container_type_objs) > 0, f"no match found {self.object_to_cook.name}"

    # pick 1 at random
    self.container = np.random.choice(container_type_objs)

    self._task_objects += [self.container]

    return self.task_rep.replace(
      'x', self.object_to_cook.name).replace(
      'y', self.container.name)

  def check_status(self):
    _, cooked = super(PlaceCookedTask, self).check_status()
    if self.container.contains:
        placed = self.container.contains.type == self.object_to_cook.type
        done = reward = cooked and placed
    else:
        done = reward = False
        placed = False
    if self.verbosity:
      print(f"cooked={cooked}, placed={placed}")
    return reward, done

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.object_to_cook, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.object_to_cook_with, actions=['place', 'pickup_container']),
      ActionsSubgoal(
        goto=self.object_to_cook_on, actions=['place', 'toggle', 'pickup_contents']),
      ActionsSubgoal(
        goto=self.container, actions=['place']),
    ]

# -----------------------
# Length 5
# -----------------------
class CookWithCleanedTask(CookTask):
  """docstring for PlaceCookedTask"""

  @property
  def task_name(self): return 'cook_with_cleaned'
  @property
  def default_task_rep(self): return 'cook x with cleaned y on z'

  def generate(self, exclude, argops=None):
    # generates x, y, z
    super(CookWithCleanedTask, self).generate(exclude, argops)

    self.object_to_cook_with.set_prop('dirty', True)

    self.sink = self.kitchen.objects_by_type(["sink"])[0]
    self.sink.set_prop('on', False)

    self._task_objects += [self.sink]

    task = self.task_rep.replace(
      'x', self.object_to_cook.name).replace(
      'y', self.object_to_cook_with.name).replace(
      'z', self.object_to_cook_on.name)
    return task


  def check_status(self):
    cleaned = self.object_to_cook_with.state['dirty'] == False
    _, cooked = super(CookWithCleanedTask, self).check_status()
    if self.verbosity:
      print(f"cooked={cooked}, cleaned={cleaned}")

    if cooked and not cleaned:
      done = True
      reward = False
    else:
      done = reward = cooked and cleaned

    return reward, done

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.object_to_cook_with, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.sink, actions=['place', 'toggle', 'pickup_contents']),
      ActionsSubgoal(
        goto=self.object_to_cook_on, actions=['place']),
      ActionsSubgoal(
        goto=self.object_to_cook, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.object_to_cook_on, actions=['place', 'toggle']),
      ]

class CookSlicedTask(CookTask):
  """docstring for PlaceCookedTask"""

  @property
  def task_name(self): return 'cook_sliced'
  @property
  def default_task_rep(self): return 'cook sliced x with y on z'

  def generate(self, exclude, argops=None):
    # generates x, y, z
    super(CookSlicedTask, self).generate(exclude, argops)

    self.knife = self.kitchen.objects_by_type(["knife"])[0]
    self._task_objects += [self.knife]

    task = self.task_rep.replace(
      'x', self.object_to_cook.name).replace(
      'y', self.object_to_cook_with.name).replace(
      'z', self.object_to_cook_on.name)
    return task


  def check_status(self):
    _, cooked = super(CookSlicedTask, self).check_status()
    sliced = self.object_to_cook.state['sliced'] == True

    if cooked and not sliced:
      done = True
      reward = False
    else:
      done = reward = cooked and sliced
    if self.verbosity:
      print(f"cooked={cooked}, sliced={sliced}")

    return reward, done

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.knife, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.object_to_cook, actions=['slice', *(['left', 'place']*4), 'pickup_contents']),
      ActionsSubgoal(
        goto=self.object_to_cook_with, actions=['place', 'pickup_container']),
      ActionsSubgoal(
        goto=self.object_to_cook_on, actions=['place', 'toggle'])
    ]

class CookSlicedWithCleanedTask(CookTask):
  """docstring for PlaceCookedTask"""

  @property
  def task_name(self): return 'cook_sliced_with_cleaned'
  @property
  def default_task_rep(self): return 'cook sliced x with cleaned y'

  def generate(self, exclude, argops=None):
    # generates x, y, z
    super(CookSlicedWithCleanedTask, self).generate(exclude, argops)

    # -----------------------
    # add slice obj
    # -----------------------
    self.knife = self.kitchen.objects_by_type(["knife"])[0]
    self._task_objects += [self.knife]


    # -----------------------
    # add clean obj
    # -----------------------
    self.object_to_cook_with.set_prop('dirty', True)
    self.sink = self.kitchen.objects_by_type(["sink"])[0]
    self.sink.set_prop('on', False)

    self._task_objects += [self.sink]


    task = self.task_rep.replace(
      'x', self.object_to_cook.name).replace(
      'y', self.object_to_cook_with.name).replace(
      'z', self.object_to_cook_on.name)
    return task


  def check_status(self):
    _, cooked = super(CookSlicedWithCleanedTask, self).check_status()
    sliced = self.object_to_cook.state['sliced'] == True
    cleaned = self.object_to_cook_with.state['dirty'] == False

    if cooked and (not sliced or not cleaned):
      done = True
      reward = False
    else:
      done = reward = cooked and sliced and cleaned
    if self.verbosity:
      print(f"cooked={cooked}, sliced={sliced}, cleaned={cleaned}")

    return reward, done

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.object_to_cook_with, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.sink, actions=['place', 'toggle', 'pickup_contents']),
      ActionsSubgoal(
        goto=self.object_to_cook_on, actions=['place']),
      ActionsSubgoal(
        goto=self.knife, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.object_to_cook, actions=['slice', *(['left', 'place']*4), 'pickup_contents']),
      ActionsSubgoal(
        goto=self.object_to_cook_on, actions=['place', 'toggle'])
    ]

class SliceAndCleanKnifeTask(KitchenTask):
  """docstring for SliceTask"""
  def __init__(self, *args, **kwargs):

    self.clean_task = CleanTask(*args, init=False, **kwargs)
    self.slice_task = SliceTask(*args, init=False, **kwargs)
    super(SliceAndCleanKnifeTask, self).__init__(*args, **kwargs)

  @property
  def task_name(self): return 'slice_and_clean_knife'
  @property
  def default_task_rep(self):
    part1 = self.slice_task.default_task_rep.replace("x", "y")
    part2 = self.clean_task.default_task_rep
    return f"{part1} and {part2}"

  def generate(self, exclude=[], argops=None):
    if argops:
      raise NotImplementedError
    slice_instr = self.slice_task.generate(
      argops=self.argument_options.get('y', None))
    clean_instr = self.clean_task.generate(argops=['knife'])

    self._task_objects = self.slice_task.task_objects + [self.clean_task.sink]

    self.slice_task.knife.set_prop('dirty', False)

    instr =  self.task_rep.replace(
      'x', self.clean_task.task_objects[0].name).replace(
      'y', self.slice_task.task_objects[0].name)

    return instr

  @property
  def num_navs(self): return 2

  def check_status(self):
    _, clean = self.clean_task.check_status()
    _, sliced = self.slice_task.check_status()

    if self.use_subtasks:
      reward = float(clean) + float(sliced)
      done = clean and sliced
    else:
      reward = done = clean and sliced

    return reward, done

  def subgoals(self):
    subgoals = self.slice_task.subgoals() + self.clean_task.subgoals()
    return subgoals

# ======================================================
# Compositions
# ======================================================
class CompositionClass(KitchenTask):
  """docstring for CompositionClass"""
  def __init__(self, *args, classes, **kwargs):
    self.classes = []
    self.variables = ['x', 'y', 'z', 'a', 'b', 'c', 'u', 'w', 'v']
    self.variables += [v.upper() for v in self.variables]
    new_kwargs = {**kwargs}
    new_kwargs['init'] = False
    for c in classes:
      Cls = c(*args, **new_kwargs)
      self.classes.append(Cls)
    super(CompositionClass, self).__init__(*args, **kwargs)


  @property
  def task_name(self): 
    names = [c.task_name for c in self.classes]
    name = "_and_".join(names)
    return name

  @property
  def default_task_rep(self):
    names = [c.default_task_rep for c in self.classes]
    new_names = []
    idx = 0
    for name in names:
      variables = [x for x in name.split(" ") if x in self.variables]
      for v in variables:
        name = name.replace(v, self.variables[idx])
        idx += 1
      new_names.append(name)

    name = " and ".join(new_names)

    return name

  def generate(self, exclude=[], argops=None):
    if argops:
      raise NotImplementedError

    task_types = []
    self._task_objects = set()
    instrs = []
    for idx, c in enumerate(self.classes):
      instr = c.generate(
        exclude=task_types,
        argops=self.argument_options.get('y', None))
      task_types.extend(c.task_types)
      self._task_objects.update(c.task_objects)
      instrs.append(instr)
    overall_instr = " and ".join(instrs)
    return overall_instr

  @property
  def num_navs(self): return len(self.classes)

  def place_object(self, object):
    room = self.env.get_room(0, 0)
    pos = self.env.place_obj(
        object,
        room.top,
        room.size,
        reject_fn=reject_next_to,
        max_tries=1000)

  def remove_object(self, _object):
    # reset position
    pos = _object.cur_pos
    if (pos >= 0).all():
      self.env.grid.set(*pos, None)

    if _object == self.env.carrying:
      self.env.carrying = None
      self.kitchen.update_carrying(None)

  def reset_on_subtask_end(self, subtask):
    """Either remove unique objects not shared across tasks or respawn all
    subtask objects.
    
    Args:
        subtask (TYPE): Description
    """
    subtask_objects = set(subtask.task_objects)


    if self.reset_behavior == 'none':
      pass
    elif self.reset_behavior == 'remove':
      # unique task objects
      # e.g. other=(knife, apple), subtask=(knife, orange)
      # shared = knife
      shared = subtask_objects
      for c in self.classes:
        shared = shared.intersection(c.task_objects)

      # nontask = apple
      nontask =  self._task_objects - subtask_objects

      # unique = (knife, orange) - apple - knife = orange
      unique = subtask_objects - nontask - shared


      # remove orange
      for _object in unique:
        self.remove_object(_object)
      for _object in shared:
        _object.contains = None

    elif self.reset_behavior == 'remove_all':
      # remove orange
      for _object in subtask_objects:
        _object.contains = None
      for _object in subtask_objects:
        self.remove_object(_object)

    elif self.reset_behavior == 'respawn':
      subtask.reset_task()
      # all task objects

      for _object in subtask_objects:
        self.remove_object(_object)
        self.place_object(_object)

  def check_and_update_status(self):
    if self.use_subtasks:
      num_done = 0
      reward = 0
      for c in self.classes:
        sub_reward, sub_done = c.check_and_update_status()
        reward += float(sub_reward)
        num_done += int(sub_done)

        if sub_done:
          self.reset_on_subtask_end(c)
      done = num_done == len(self.classes)
    else:
      goals_achieved = True
      for c in self.classes:
        _, sub_done = c.check_and_update_status()
        if sub_done:
          self.reset_on_subtask_end(c)
        goals_achieved = goals_achieved and c.finished

      reward, done = self.update_status(goals_achieved)

    if self.verbosity:
      print(f"{self.task_name}: reward={reward}, done={done}")
      print(f"\ttime_complete={c._time_complete}, goals_achieved={goals_achieved}")
      print(f"\tsub_done={sub_done}, finished={c.finished}")

    return reward, done

  def subgoals(self):
    subgoals = []
    for c in self.classes:
      subgoals.extend(c.subgoals())
    return subgoals

Slice2Task = functools.partial(CompositionClass, classes=[SliceTask, SliceTask])
Slice3Task = functools.partial(CompositionClass, classes=[SliceTask, SliceTask, SliceTask])
Toggle2Task = functools.partial(CompositionClass, classes=[ToggleTask, ToggleTask])
Toggle3Task = functools.partial(CompositionClass, classes=[ToggleTask, ToggleTask, ToggleTask])
Clean2Task = functools.partial(CompositionClass, classes=[CleanTask, CleanTask])
Clean3Task = functools.partial(CompositionClass, classes=[CleanTask, CleanTask, CleanTask])
Cook2Task = functools.partial(CompositionClass, classes=[CookTask, CookTask])
Cook3Task = functools.partial(CompositionClass, classes=[CookTask, CookTask, CookTask])


CleanAndSliceTask = functools.partial(CompositionClass, classes=[SliceTask, CleanTask])
ToggleAndSliceTask = functools.partial(CompositionClass, classes=[ToggleTask, SliceTask])
ToggleAndPickupTask = functools.partial(CompositionClass, classes=[PickupTask, ToggleTask])
CleanAndToggleTask  = functools.partial(CompositionClass, classes=[CleanTask, ToggleTask])
CleanAndSliceAndToggleTask = functools.partial(CompositionClass, classes=[SliceTask, CleanTask, ToggleTask])

CookAndSliceTask = functools.partial(CompositionClass, classes=[CookTask, SliceTask])


def get_class_name(name: str):
  return name.title().replace(" ","") + "Task"

def make_class(name: str, negate: bool=False):
  numbers = re.findall(r'\d+', name)
  assert len(numbers) in [0,1]
  if len(numbers) > 0:
    name = name.replace(numbers[0], "")

  class_name = get_class_name(name)
  current_module = sys.modules[__name__]
  try:
    Cls = getattr(current_module, class_name)
  except Exception as e:
    raise f"{current_module} not module"

  if len(numbers) > 0:
    return functools.partial(CompositionClass, classes=[Cls]*int(numbers[0]), negate=negate)
  else:
    return functools.partial(Cls, negate=negate)

def make_composite(name: str):

  # clean_and_not_toggle --> [clean, not_toggle]
  bases = name.split("_and_")

  # [False, True]
  negate = ["not" in b for b in bases]

  # [clean, not_toggle] --> [clean, toggle]
  clean_bases = [b.replace("not_", "").replace("_", " ") for b in bases]
  classes = [make_class(b, n) for b, n in zip(clean_bases, negate)]

  return functools.partial(CompositionClass, classes=classes)

def get_task_class(name, only_composite=False):
  if not only_composite:
    defaults = all_tasks()
    if name in defaults:
      return defaults[name]

  if 'and' in name:
    return make_composite(name)
  else:
    return make_composite(name)

def all_tasks():
  return dict(
    pickup=PickupTask,
    toggle=ToggleTask,
    place=PlaceTask,
    heat=HeatTask,
    clean=CleanTask,
    slice=SliceTask,
    slice_putdown=SlicePutdownTask,
    chill=ChillTask,
    slice2=Slice2Task,
    cook2=Cook2Task,
    cook3=Cook3Task,
    toggle2=Toggle2Task,
    slice3=Slice3Task,
    toggle3=Toggle3Task,
    clean2=Clean2Task,
    clean_and_slice=CleanAndSliceTask,
    slice_and_clean_knife=SliceAndCleanKnifeTask,
    clean_and_toggle=CleanAndToggleTask,
    toggle_and_slice=ToggleAndSliceTask,
    toggle_and_pickup=ToggleAndPickupTask,
    clean_and_slice_and_toggle=CleanAndSliceAndToggleTask,
    cook_and_slice=CookAndSliceTask,
    pickup_cleaned=PickupCleanedTask,
    pickup_sliced=PickupSlicedTask,
    pickup_chilled=PickupChilledTask,
    pickup_heated=PickupHeatedTask,
    cook=CookTask,
    pickup_cooked=PickupCookedTask,
    pickup_placed=PickupPlacedTask,
    place_sliced=PlaceSlicedTask,
)
