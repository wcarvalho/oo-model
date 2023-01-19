import logging
import numpy as np
import collections
from gym import spaces
from gym.utils import seeding
from pprint import pprint

from gym_minigrid.minigrid import Grid, WorldObj
from babyai.levels.levelgen import RoomGridLevel, RejectSampling


from envs.babyai_kitchen.world import Kitchen
import envs.babyai_kitchen.tasks


TILE_PIXELS = 32

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def reject_next_to(env, pos):
  """
  Function to filter out object positions that are right next to
  the agent's starting point
  """

  sx, sy = env.agent_pos
  x, y = pos
  d = abs(sx - x) + abs(sy - y)
  return d < 2

class KitchenLevel(RoomGridLevel):
  """
  """
  def __init__(
    self,
    room_size=8,
    num_rows=1,
    num_cols=1,
    num_dists=0,
    debug=False,
    # locked_room_prob=0,
    unblocking=False,
    kitchen=None,
    random_object_state=False,
    objects = [],
    actions = ['left', 'right', 'forward', 'pickup_container', 'pickup_contents', 'place', 'toggle', 'slice'],
    load_actions_from_tasks=False,
    task_kinds=['slice', 'clean', 'cook'],
    valid_tasks=[],
    taskarg_options=None,
    task_reps=None,
    instr_kinds=['action'], # IGNORE. not implemented
    use_subtasks=False,
    task_reset_behavior: str='none',
    use_time_limit=True,
    tile_size=8,
    rootdir='.',
    distant_vision=False,
    agent_view_size=7,
    reward_coeff=1.0,
    extra_timesteps=1,
    seed=None,
    verbosity=0,
    nseeds=500,
    **kwargs,
      ):
    self.num_dists = num_dists
    self.debug = debug
    self.nseeds = nseeds
    # self.locked_room_prob = locked_room_prob
    self.use_time_limit = use_time_limit
    self.unblocking = unblocking
    self.valid_tasks = list(valid_tasks)
    if isinstance(task_kinds, list):
        self.task_kinds = task_kinds
    elif isinstance(task_kinds, str):
        self.task_kinds = [task_kinds]
    else:
        RuntimeError(f"Don't know how to read task kind(s): {str(task_kinds)}")
    self.task_reps = task_reps or dict()
    self.instr_kinds = instr_kinds
    self.random_object_state = random_object_state
    self.task_reset_behavior = task_reset_behavior.lower()
    if self.task_reset_behavior == 'respawn':
      if not use_subtasks:
        logging.info("Turning subtasks on for object respawning behavior")
      use_subtasks = True
    self.use_subtasks = use_subtasks

    self.taskarg_options = taskarg_options
    self.reward_coeff = reward_coeff
    self.verbosity = verbosity
    self.locked_room = None
    self.extra_timesteps = extra_timesteps

    assert room_size >= 5, "otherwise can never place objects"
    agent_view_size = min(agent_view_size, room_size)
    if agent_view_size % 2 !=1:
        agent_view_size -= 1
    # ======================================================
    # agent view
    # ======================================================
    self.agent_view_width = agent_view_size
    if distant_vision:
        raise NotImplementedError
        # for far agent can see is length of room (-1 for walls)
        self.agent_view_height = room_size - 1
    else:
        self.agent_view_height = agent_view_size

    # ======================================================
    # setup env
    # ======================================================
    # define the dynamics of the objects with kitchen

    self.kitchen = kitchen or Kitchen(objects=objects, tile_size=tile_size, rootdir=rootdir, verbosity=verbosity)
    self.check_task_actions = False

    # to avoid checking task during reset of initialization
    super().__init__(
        room_size=room_size,
        num_rows=num_rows,
        num_cols=num_cols,
        seed=seed,
        agent_view_size=self.agent_view_height,
        **kwargs,
    )
    self.check_task_actions = True

    # ======================================================
    # action space
    # ======================================================
    self.actiondict = {action:idx for idx, action in enumerate(actions, start=0)}

    # -----------------------
    # for backward compatibility
    # below is used by parent classes
    # -----------------------
    pickup_idx = self.actiondict['pickup_contents']
    backwards = {
      'pickup': pickup_idx,
      'done': len(self.actiondict)
      }
    candrop = 'place' in self.actiondict
    if candrop:
      backwards['drop'] =  self.actiondict['place']

    # ActionCls = collections.namedtuple('Action', actions 
    #   + ['pickup', 'done'] 
    #   + (['drop'] if candrop else []))

    backwards_action_dict = {**self.actiondict, **backwards}
    self.actions = AttrDict(**backwards_action_dict)

    # -----------------------
    # below is used by this class
    # -----------------------
    self.idx2action = {idx:action for idx, action in enumerate(actions, start=0)}
    self.action_names = actions
    naction = len(self.actiondict)
    self.action_space = spaces.Discrete(naction)


    # ======================================================
    # observation space
    # ======================================================
    # potentially want to keep square and just put black for non-visible?
    shape=(self.agent_view_height, self.agent_view_width, 3)


    self.observation_space.spaces['image'] = spaces.Box(
        low=0,
        high=self.kitchen._max_states,
        shape=shape,
        dtype='int32'
    )


  # ======================================================
  # functions for generating grid + objeccts
  # ======================================================
  def _gen_grid(self, *args, **kwargs):
    """dependencies between RoomGridLevel, MiniGridEnv, and RoomGrid are pretty confusing so just call base _gen_grid function to generate grid.
    """
    super(RoomGridLevel, self)._gen_grid(*args, **kwargs)

  def place_in_room(self, i, j, obj):
      """
      Add an existing object to room (i, j)
      """

      room = self.get_room(i, j)

      pos = self.place_obj(
          obj,
          room.top,
          room.size,
          reject_fn=reject_next_to,
          max_tries=1000
      )

      room.objs.append(obj)

      return obj, pos

  def add_objects(self, task=None, num_distractors=10):
    """
    - if have task, place task objects
    
    Args:
        task (None, optional): Description
        num_distactors (int, optional): Description
    """
    placed_objects = set()
    # first place task objects
    if task is not None:
        for obj in task.task_objects:
            self.place_in_room(0, 0, obj)
            placed_objects.add(obj)
            if self.verbosity > 1:
                print(f"Added task object: {obj}")

    # if number of left over objects is less than num_distractors, set as that
    # possible_space = (self.grid.width - 2)*(self.grid.height - 2)
    num_leftover_objects = len(self.kitchen.objects)-len(placed_objects)
    num_distractors = min(num_leftover_objects, num_distractors)

    if len(placed_objects) == 0:
        num_distractors = max(num_distractors, 1)

    self.distractors_added = []
    num_tries = 0

    while len(self.distractors_added) < num_distractors:
        # infinite loop catch
        num_tries += 1
        if num_tries > 1000:
            raise RuntimeError("infinite loop in `add_objects`")

        # sample objects
        random_object = np.random.choice(self.kitchen.objects)

        # if already added, try again
        if random_object in placed_objects:
            continue

        self.distractors_added.append(random_object)
        placed_objects.add(random_object)

    if self.verbosity > 0:
      print('-'*10)
      print("Distractors:", [d.type for d in self.distractors_added])

    for random_object in self.distractors_added:
        random_object.reset()
        self.place_in_room(0, 0, random_object)
        if self.verbosity > 0:
          print('-'*10)
          print(f"Added distractor: {random_object.type}")
          room = self.get_room(0, 0)
          pprint([o.type for o in room.objs])

    # TODO: test ``active objects''
    # self.kitchen.set_active_objects(placed_objects)
  
  # ======================================================
  # functions for generating and validating tasks
  # ======================================================
  def rand_task(
    self,
    task_kinds,
    instr_kinds=['action'],
    use_subtasks=False,
    only_composite=False,
    **kwargs):
    """Summary
    
    Args:
        task_kinds (TYPE): Description
        instr_kinds (list, optional): Description
        use_subtasks (bool, optional): Description
        only_composite (bool, optional): only use `CompositionClass` which supports special reset behavior
        **kwargs: Description
    
    Returns:
        TYPE: Description
    
    Raises:
        RuntimeError: Description
    """
    instruction_kind = np.random.choice(instr_kinds)

    if instruction_kind == 'action':
        if isinstance(task_kinds, list):
          task_kind = np.random.choice(task_kinds).lower()
        elif isinstance(task_kinds, str):
          task_kind = task_kinds
        else:
          raise RuntimeError
        if task_kind == 'none':
            task = None
        else:
            # available_tasks = envs.babyai_kitchen.tasks.all_tasks()
            # task_class = available_tasks[task_kind]
            task_class = envs.babyai_kitchen.tasks.get_task_class(task_kind, only_composite=only_composite)
            task = task_class(
                env=self,
                kitchen=self.kitchen,
                task_reps=self.task_reps,
                done_delay=self.extra_timesteps,
                reset_behavior=self.task_reset_behavior,
                verbosity=self.verbosity,
                use_subtasks=use_subtasks,
                **kwargs)
    else:
        raise RuntimeError(f"Instruction kind not supported: '{instruction_kind}'")

    return task

  def generate_task(self):
    """copied from babyai.levels.levelgen:LevelGen.gen_mission
    """

    # connect all rooms
    self.connect_all()

    # reset kitchen objects
    self.kitchen.reset(randomize_states=self.random_object_state)

    # Generate random instructions
    task = self.rand_task(
        task_kinds=self.task_kinds,
        argument_options=self.taskarg_options,
        instr_kinds=self.instr_kinds,
        use_subtasks=self.use_subtasks,
    )
    if self.valid_tasks:
        idx = 0
        while not task.instruction in self.valid_tasks:
            task = self.rand_task(
                task_kinds=self.task_kinds,
                argument_options=self.taskarg_options,
                instr_kinds=self.instr_kinds,
                use_subtasks=self.use_subtasks,
            )
            idx += 1
            if idx > 1000:
                raise RuntimeError("infinite loop sampling possible task")


    self.add_objects(task=task, num_distractors=self.num_dists)

    # The agent must be placed after all the object to respect constraints
    while True:
        self.place_agent()
        start_room = self.room_from_pos(*self.agent_pos)
        # Ensure that we are not placing the agent in the locked room
        if start_room is self.locked_room:
            continue
        break

    # self.unblocking==True means agent may need to unblock. don't check
    # self.unblocking==False means agent does not need to unblock. check
    if not self.unblocking:
        self.check_objs_reachable()


    return task

  def validate_task(self, task):
    if task is not None and self.check_task_actions:
        task.check_actions(self.action_names)

  def reset_task(self):
    """copied from babyai.levels.levelgen:RoomGridLevel._gen_drid
    - until success:
        - generate grid
        - generate task
            - generate objects
            - place object
            - generate language instruction
        - validate instruction
    """
    # We catch RecursionError to deal with rare cases where
    # rejection sampling gets stuck in an infinite loop
    tries = 0
    while True:
        if tries > 1000:
            raise RuntimeError("can't sample task???")
        try:
            tries += 1
            if self.verbosity > 0:
              print(f"RESET ATTEMPT {tries}")
            # generate grid of observation
            self._gen_grid(width=self.width, height=self.height)

            # Generate the task
            task = self.generate_task()

            # Validate the task
            self.validate_task(task)


        except RecursionError as error:
            print(f'Timeout during mission generation:{tries}/1000\n', error)
            continue

        except RejectSampling as error:
            #print('Sampling rejected:', error)
            continue

        break

    return task

  # ======================================================
  # reset, step used by gym
  # ======================================================
  def reset(self, **kwargs):
    if self.nseeds:
      seeding.np_random(self.nseeds)
    """Copied from: 
    - gym_minigrid.minigrid:MiniGridEnv.reset
    - babyai.levels.levelgen:RoomGridLevel.reset
    the dependencies between RoomGridLevel, MiniGridEnv, and RoomGrid were pretty confusing so I rewrote the base reset function.
    """
    # ======================================================
    # copied from: gym_minigrid.minigrid:MiniGridEnv.reset
    # ======================================================
    # reset current position and direction of the agent
    self.agent_pos = None
    self.agent_dir = None

    # -----------------------
    # generate:
    # - grid
    # - objects
    # - agent location
    # - instruction
    # -----------------------
    # when call reset during initialization, don't load
    self.kitchen.reset()
    self.task = self.reset_task()
    if self.task is not None:
        self.surface = self.task.surface(self)
        self.mission = self.surface
        self.instrs = self.task


        # make sure all task objects are on grid
        for obj in self.task.task_objects:
            assert obj.init_pos is not None
            assert obj.cur_pos is not None
            assert np.all(obj.init_pos == obj.cur_pos)
            assert self.grid.get(*obj.init_pos) is not None

    else:
        self.surface = self.mission = "No task"

    # These fields should be defined by _gen_grid
    assert self.agent_pos is not None
    assert self.agent_dir is not None

    # Check that the agent doesn't overlap with an object
    start_cell = self.grid.get(*self.agent_pos)
    assert start_cell is None or start_cell.can_overlap()

    # Item picked up, being carried, initially nothing
    self.carrying = None

    # Step count since episode start
    self.step_count = 0

    # Return first observation
    obs = self.gen_obs()

    # updating carrying in kitchen env just in case
    self.kitchen.update_carrying(self.carrying)
    # ======================================================
    # copied from babyai.levels.levelgen:RoomGridLevel.reset
    # ======================================================
    if self.task is not None:
        reward, done = self.task.check_status()
        if done:
            raise RuntimeError(f"`{self.mission}` started off as done")

    # Compute the time step limit based on the maze size and instructions
    nav_time_room = int(self.room_size ** 2)
    nav_time_maze = nav_time_room * self.num_rows * self.num_cols
    if self.task:
        num_navs = self.task.num_navs
    else:
        num_navs = 2
    self.max_steps = max(2, num_navs) * nav_time_maze
    self.timesteps_complete = 0
    self.interaction_info = {}

    if self.debug:
      self.max_steps = max(np.random.randint(5), 1)

    return obs

  def straction(self, action : str):
    return self.actiondict[action]

  def step(self, action):
    """Copied from: 
    - gym_minigrid.minigrid:MiniGridEnv.step
    - babyai.levels.levelgen:RoomGridLevel.step
    This class derives from RoomGridLevel. We want to use the parent of RoomGridLevel for step. 
    """
    # ======================================================
    # copied from MiniGridEnv
    # ======================================================
    self.step_count += 1

    reward = 0
    done = False

    # Get the position in front of the agent
    fwd_pos = self.front_pos

    # Get the contents of the cell in front of the agent
    object_infront = self.grid.get(*fwd_pos)


    # Rotate left
    action_info = None
    self.interaction_info = {}
    interaction = False
    if action == self.actiondict.get('left', -1):
        self.agent_dir -= 1
        if self.agent_dir < 0:
            self.agent_dir += 4

    # Rotate right
    elif action == self.actiondict.get('right', -1):
        self.agent_dir = (self.agent_dir + 1) % 4

    # Move forward
    elif action == self.actiondict.get('forward', -1):
        if object_infront == None or object_infront.can_overlap():
            self.agent_pos = fwd_pos
        # if object_infront != None and object_infront.type == 'goal':
        #     done = True
        #     reward = self._reward()
        # if object_infront != None and object_infront.type == 'lava':
        #     done = True
    else:
        try:
          action_info = self.kitchen.interact(
              action=self.idx2action[int(action)],
              object_infront=object_infront,
              fwd_pos=fwd_pos,
              grid=self.grid,
              env=self, # only used for backwards compatibility with toggle
          )
          if action_info['success']:
            self.interaction_info = dict(
              action=str(self.idx2action[int(action)]),
              object=str(object_infront.type) if object_infront else None)
          self.carrying = self.kitchen.carrying
          interaction = True
        except Exception as e:
          print("Action:", int(action))
          print("Actions available:", self.idx2action)
          print("Object in front:", object_infront)
          raise e

    step_info = self.kitchen.step()

    if self.verbosity > 0:
        from pprint import pprint
        print('='*50)
        obj_type = object_infront.type if object_infront else None
        action_str = self.idx2action[int(action)]
        print(f"ACTION: ({action_str}, {obj_type})", obj_type)
        if self.verbosity > 1:
          pprint(action_info)
          print('-'*10, 'Env Info', '-'*10)
          print("Carrying:", self.carrying)
          if self.task is not None:
              print(f"task objects:")
              pprint(self.task.task_objects)
          else:
              print(f"env objects:")
              pprint(self.kitchen.objects)

    # ======================================================
    # copied from RoomGridLevel
    # ======================================================

    # If we've successfully completed the mission
    info = {'success': False}
    done = False
    if self.task is not None:
        # reward, done = self.task.get_reward_done()
        reward, done = self.task.check_and_update_status()
        reward = float(reward)
        if self.verbosity > 0:
          if reward !=0:
            print("REWARD:", reward)
    # if past step count, done
    if self.step_count >= self.max_steps and self.use_time_limit:
        done = True

    obs = self.gen_obs()
    if self.debug:
      reward = 1.0

    reward = reward*self.reward_coeff
    return obs, reward, done, info



  # ======================================================
  # rendering functions
  # ======================================================

  def get_obs_render(self, obs, tile_size=TILE_PIXELS//2):
    """
    Render an agent observation for visualization
    """

    width, height, channels = obs.shape
    assert channels == 3

    vis_mask = np.ones(shape=(width, height), dtype=np.bool_)

    grid = Grid(width, height)
    for i in range(width):
        for j in range(height):
            obj_idx, color_idx, state = obs[i, j]
            if obj_idx < 11:
                object = WorldObj.decode(obj_idx, color_idx, state)
                # vis_mask[i, j] = (obj_idx != OBJECT_TO_IDX['unseen'])
            else:
                object = self.kitchen.objectid2object.get(obj_idx, None)
            if object:
                grid.set(i, j, object)





    # Render the whole grid
    img = grid.render(
        tile_size,
        agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
        agent_dir=3,
        highlight_mask=vis_mask
    )

    return img


