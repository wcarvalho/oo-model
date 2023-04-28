"""
Tasks to make (1 at a time):
  1: Pick Up X
  2: Place X on Y

Possible actions:
  1. Navigation actions: MoveAhead, MoveBack, MoveLeft, MoveRight, RotateLeft, RotateRight
  2. Interaction actions: Pickup Apple, Pickup Eggs, Pickup Milk, et.c
"""
from typing import NamedTuple
from PIL.Image import LINEAR
from ai2thor.build import CloudRendering
from ai2thor.controller import Controller
import dm_env
import matplotlib.pyplot as plt
from dm_env import specs
import random
import argparse
from typing import NamedTuple

from tensorflow.python.ops.numpy_ops import positive
from acme import types
import tree
import numpy as np
import jax.numpy as jnp
import dataclasses
import typing
import enum
import pdb
import jax
from absl import logging
from utils import check_receptacle_clean,Navigation_action,basic_vector,Interaction_action, simple_basic_vector
import time
import random

class Ai2ThorTask():
    """Task objects that generates task and checks progress."""

    def __init__(self, controller: Controller, keep_obj=None, obj_randomize=None, agent_randomize=None, mode="arm"):
        self.controller = controller
        self.keep_obj = keep_obj
        self.obj_randomize = obj_randomize
        self.agent_randomize = agent_randomize
        self.event = None
        self.action_str =None
        self.object_type = None
        self.object_id = None
        self.random_seed = 200
        self.receptacle_type = None
        self.mode = mode
    def reset(self):
        """
        Use the current state of the controller to generate a new task.
        This task object can modify the scene in the controller in order to define the task it needs.
        Maybe task = place x in fridge and there's an easy mode.
        And during easy, fridge is open. This class can change the fridge to be open then.

        """
        pass

    def get_reward(self):
        """
        Use the current state of the controller to compute reward.
        """
        pass


class PickupTask(Ai2ThorTask):

    def __init__(self, *args, **kwargs):

        super(PickupTask, self).__init__(*args, **kwargs)

    def reset(self):

        self.done = False
        self.reward = 0.0
        self.max_step_count = 100
        self.step_count = 0

        if self.obj_randomize:
            random_seed = random.randrange(self.random_seed)
            self.controller.step(
                action="InitialRandomSpawn",
                randomSeed=random_seed,
                forceVisible=False,
                numPlacementAttempts=30,
                placeStationary=True
            )


        if self.agent_randomize:
            # get the position
            positions = self.controller.step(
                action="GetReachablePositions"
            ).metadata["actionReturn"]

            # teleport to a position
            position = random.choice(positions)
            self.event = self.controller.step(
                action="Teleport",
                position=position
            )

        all_objs = []
        """if env has N objects. Remove all except what's specified by `keep'."""
        if self.keep_obj:
            for obj in self.controller.last_event.metadata["objects"]:
                if obj["objectType"] not in self.keep_obj and obj["pickupable"]:
                    self.controller.step(
                        action="RemoveFromScene",
                        objectId=obj["objectId"]
                    )
                else:
                    # only append the pickupable object
                    if obj["pickupable"]:
                        all_objs.append(obj["objectType"])
        else:
            for obj in self.controller.last_event.metadata["objects"]:
                # only append the pickupable object
                if obj["pickupable"]:
                    all_objs.append(obj["objectType"])

        """collect all object and store into a list with (type, id) tuple"""
        random_obj = all_objs[random.randint(0, len(all_objs)-1)]

        ## move agent next to the Tomato
        self.controller.step(action = "LookDown")
        self.controller.step(action="MoveRight")
        self.controller.step(action="MoveAhead")
        #self.controller.step(action="MoveArmBase", y=0.6)
        self.controller.step(action="MoveArm", position=dict(x=0,y=0.1,z=0), coordinateSpace="wrist")
        self.controller.step(action="MoveArm", position=dict(x=0,y=0,z=0.1), coordinateSpace="wrist")
        #self.controller.step(action="MoveArm", position=dict(x=0,y=0,z=0.1), coordinateSpace="wrist")

        ''' mapping action, object into integer'''
        # action_str: PickupObject
        # object_type: e.g: Apple
        self.action_str = Action.PickupObject
        self.object_type = "Tomato"
        #pdb.set_trace()

    def get_reward(self):
        """
        Use the current state of the controller to compute reward.
        return reward and done mark
        """
        if self.step_count < self.max_step_count :
            self.done = True

        self.step_count += 1
        # Handle ManipulaTHOR
        if self.mode == "arm":

            hold_objs = self.controller.last_event.metadata["arm"]["heldObjects"]
            if len(hold_objs) != 0:
                # since it may not be just hold one object
                for i in range(len(hold_objs)):
                    tokens = hold_objs[i].split("|")
                    if tokens[0] == self.object_type:

                        self.reward = 1.0
        # Handle iTHOR
        else:
            for obj in self.controller.last_event.metadata["objects"]:
                if obj["objectType"] == self.object_type:
                    if obj["isPickedUp"]:
                        self.reward = 1.0
        return self.done, self.reward

class PlaceTask(Ai2ThorTask):

    def __init__(self, *args, **kwargs):
        super(PlaceTask, self).__init__(*args, **kwargs)

    def reset(self):

        self.done = False
        self.reward = 0.0
        self.max_step_count = 100
        self.step_count = 0

        self.controller.step(action="SetHandSphereRadius",
                            radius=0.1
                            )

        if self.obj_randomize:
            random_seed = random.randrange(self.random_seed)
            self.controller.step(
                action="InitialRandomSpawn",
                randomSeed=random_seed,
                forceVisible=False,
                numPlacementAttempts=30,
                placeStationary=True
            )


        if self.agent_randomize:
            # get the position
            positions = self.controller.step(
                action="GetReachablePositions"
            ).metadata["actionReturn"]

            # teleport to a position
            position = random.choice(positions)
            self.event = self.controller.step(
                action="Teleport",
                position=position
            )

        all_objs = []
        """if env has N objects. Remove all except what's specified by `keep'."""
        if self.keep_obj:
            for obj in self.controller.last_event.metadata["objects"]:
                if obj["objectType"] not in self.keep_obj and obj["pickupable"]:
                    self.controller.step(
                        action="RemoveFromScene",
                        objectId=obj["objectId"]
                    )
                else:
                    # only append the pickupable object
                    if obj["pickupable"]:
                        all_objs.append((obj["objectType"]))
        else:
            for obj in self.controller.last_event.metadata["objects"]:
                # only append the pickupable object
                if obj["pickupable"]:
                    all_objs.append((obj["objectType"]))

        # define the target object we want to place
        target_obj = all_objs[random.randint(0, len(all_objs)-1)]

        valid_receptacles = []
        for obj in self.controller.last_event.metadata["objects"]:
            # search and store valid receptacle
            if obj["receptacle"]:
                valid_receptacles.append(obj["objectType"])

        # define the target receptacle that the object will put in/on
        target_receptacle = valid_receptacles[random.randint(0, len(valid_receptacles)-1)]

        ## put the holding Tomato to the receptacle Pot.
        self.controller.step(action = "LookDown")
        self.controller.step(action="MoveRight")
        self.controller.step(action="MoveAhead")
        self.controller.step(action="MoveArm", position=dict(x=0,y=0.1,z=0), coordinateSpace="wrist")
        self.controller.step(action="MoveArm", position=dict(x=0,y=0,z=0.1), coordinateSpace="wrist")
        self.controller.step(action="PickupObject")
        self.controller.step(action="MoveArm", position=dict(x=-0.4,y=0.3,z=0), coordinateSpace="wrist")
        self.controller.step(action="RotateRight")
        self.controller.step(action="MoveAhead")
        self.controller.step(action="MoveAhead")
        #self.controller.step(action="ReleaseObject")

        self.action_str = Action.ReleaseObject
        #self.receptacle_type = target_receptacle
        self.receptacle_type = "Pot"
        self.target_obj = "Tomato"

        check_receptacle_clean(self.controller,self.target_obj,self.receptacle_type,self.random_seed)

    def get_reward(self):
        """
        Use the current state of the controller to compute reward.
        return reward and done mark
        """
        if self.step_count < self.max_step_count :
            self.done = True

        self.step_count += 1

        for obj in self.controller.last_event.metadata["objects"]:
            if obj["objectType"] == self.target_obj:
                parent_list = obj["parentReceptacles"]
                # iterate through the parent_rep_list to find the target rep
                # parent_list = [Stool|-02.08|+00.94|-03.62, .....]
                if parent_list is None:
                    continue
                else:
                    for i in range(len(parent_list)):
                        tokens = parent_list[i].split("|")
                        if tokens[0] == self.receptacle_type:
                            self.reward =1.0
        return self.done, self.reward

class Observation(NamedTuple):
    """Container for (image, task) tuples"""
    image: types.Nest
    mission: types.Nest

class Ai2thorEnv(dm_env.Environment):
    """Environment for Ai2THor.

    This will be used to have an agent select actions in the Ai2Thor simulator. At each step, environment will compute reward for the agent given its action.

    Main purpose of class:
      Getting correct outputs needed for RL loop.
      - reset: observation/state
      - step: observation/state, reward, done, info

      Also handle converting int actions to controller actions.
    """

    def __init__(self, controller: Controller):

        self.controller = controller
        self.keep_obj = None
        self.obj_randomize = None
        self.agent_randomize = None
        self.task_tuple = None
        self.image = None
        self.action = None
        self.discount = 0.0
        self.done =False
        self.downsample = True
        self.shape = (300,300,3)
        self.d_shape = (80,80,3)
        self.pickup_task = False
        self.place_task = True
        self.pickup_task_len = 2
        self.place_task_len = 2
        self.mode = "arm"
        self.task_string = None
        self.refresh = True
        self.init_pos = None
    def reset(self):
        """
        1. resent the controller (e.g. reset the room)
        2. generate a new task (e.g. select object as goal object for "Pickup X")

        Returns:
          dm_env.TimeStep
        """
        self.reward = False

        #Ai2ThorTask(self.controller, self.keep_obj, self.obj_randomize, self.agent_randomize)
        self.controller.reset("FloorPlan14")
        if self.pickup_task:
            self.task = PickupTask(self.controller, self.keep_obj, self.obj_randomize, self.agent_randomize, self.mode)
            self.task.reset()
            self.task_string = "PickupObject " + self.task.object_type
            self.task_tuple = np.array([Action(self.task.action_str).value, Object[self.task.object_type].value],dtype=int).reshape(2,)
        else:
            self.task = PlaceTask(self.controller, self.keep_obj, self.obj_randomize, self.agent_randomize,self.mode)
            self.task.reset()
            self.task_string = "ReleaseObject " + self.task.receptacle_type
            self.task_tuple = np.array([Action(self.task.action_str).value, Object[self.task.receptacle_type].value],dtype=int).reshape(2,)
            image = self.controller.last_event.frame
            plt.imsave("reset.png",image)
        self.image = self.controller.last_event.frame
        if self.downsample:
            self.image = jax.image.resize(image=self.image, shape=self.d_shape, method="bilinear")

        self.image = jax.tree_map(lambda x: x.astype(jnp.float32), self.image)

        if self.mode == "arm":
            self.executor = Executor(self.controller)
        else:
            self.executor = Executor_ai2thor(self.controller)

        # construct action space dynamicly before the episode start
        self.executor.create_action_space()

        return dm_env.restart(observation=Observation(
                image=self.image,
                mission=self.task_tuple,
            ))

    def step(self, action):
        """
        1. use action to update controller, test string action first, then move to int
        2. use Ai2ThorTask to see if task is complete

        Returns:
          dm_env.TimeStep
        """
        # take one step action
        self.executor.step(action)
        self.image = self.controller.last_event.frame

        # reduce the pixel from (300,300,3) ->(80,80,3)
        if self.downsample :
            self.image = jax.image.resize(image=self.image, shape=self.d_shape, method="bilinear")

        # convert uint8 to float
        self.image = jax.tree_map(lambda x: x.astype(jnp.float32), self.image)

        done, reward = self.task.get_reward()
        #check if task is complete
        if reward == 1 or done:
            return dm_env.termination(reward=reward,
                                      observation=Observation(image=self.image,mission=self.task_tuple
                                                       )
                                    )
        else:
            return dm_env.transition(reward=reward,
                                     observation=Observation(image=self.image,mission=self.task_tuple
                                                       )
                                    )

    def observation_spec(self) :
        """Returns the observation spec."""
        if self.downsample:
            self.shape = self.d_shape
        if self.pickup_task:
            self.task_len = self.pickup_task_len
        else:
            self.task_len = self.place_task_len

        return Observation(
        # switch to np.uint8 eventually
        image = specs.BoundedArray(shape=self.shape,
                                   dtype=float,
                                   name="image",
                                   minimum=0,
                                   maximum=255,
                                   ),
        mission = specs.Array(shape=(self.task_len,),
                                  dtype=int,
                                  name="mission",
                                  ),
        )
    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(
        dtype=np.int32, num_values=17, name="action")

    def reward_spec(self) -> specs.Array:
        """Returns the reward spec."""
        return specs.Array(shape=(),
        dtype=float, name="reward")

    def discount_spec(self) -> specs.Array:
        """Returns the discount spec."""
        return specs.Array(shape=(),
        dtype=float, name="discount")

class Executor(Ai2ThorTask):
    def __init__(self, controller):

        super(Executor, self).__init__(controller)
        self.controller = controller
        self.action = None
        self.IDX_TO_ACTION = {}
        self.object_str = None
        self.action_len = 0
        self.Navigation_action = Navigation_action
        self.Interaction_action =Interaction_action
        self.arm_vector = basic_vector
        self.simple_arm_vector = simple_basic_vector
        self.scale = 0.1
        self.hand_radius = 0.1
        self.interval = 0.1
    def create_action_space(self):

        # first, set the hand sphere radius to 0.5 meter
        self.controller.step(action="SetHandSphereRadius",
                            radius=self.hand_radius
                            )

        # create idx_to_action dynamicly before episode start
        action_to_idx = {}
        idx = 0
        # Handle the Arm
        for j in self.simple_arm_vector:
            action_to_idx[idx] = ("MoveArm", j)
            idx += 1
        for i in range(len(self.Interaction_action)):
            action_to_idx[idx] = (self.Interaction_action[i])
            idx += 1
        for i in range(len(self.Navigation_action)):
            action_to_idx[idx] = (self.Navigation_action[i])
            idx += 1
        self.IDX_TO_ACTION = action_to_idx

    def step(self, action):
        '''This function defines how to convert action to what's needed for Thor'''

        # convert action from integer to string e.g:9 -> PickupObject
        self.action = action.item()
        #self.action = action
        action_tuple= self.IDX_TO_ACTION[self.action]

        # Handle interactive action, like pickup, release, MoveArm
        if len(action_tuple) == 2:
            action_str, target = action_tuple
            # instead of "teleport" arm, we move arm based on the cur position
            if action_str == "MoveArm":
                self.controller.step(action=action_str, position=target, coordinateSpace="wrist")
            # Move the whole arm vertically
            elif action_str == "MoveArmBase":
                self.controller.step(action=action_str, y=target)
            # handle pickup and realese.
            else:
                self.controller.step(action=action_str)
        # Handle Navigation action
        else:
            self.controller.step(action=action_tuple)
            if action_tuple == "ReleaseObject":
                # take an extra step so that the screen can be updated to
                # the screen of falling objects.
                self.controller.step(action="MoveAhead")

class Executor_ai2thor(Ai2ThorTask):
    def __init__(self, controller):

        super(Executor_ai2thor, self).__init__(controller)
        self.controller = controller
        self.action = None
        self.IDX_TO_ACTION = {}
        self.object_str = None
        self.action_len = 0
        self.Navigation_action = Navigation_action
        self.basic_vector = basic_vector
        self.scale = 0.1
    def create_action_space(self):
        # create idx_to_action dynamicly before episode start
        action_to_idx = {}
        idx = 0
        for obj in self.controller.last_event.metadata["objects"]:

            # handle pickup object action space
            if obj["pickupable"] and ("PickupObject",obj["objectType"]) not in action_to_idx:
                # handle duplicate obj
                action_to_idx[("PickupObject",obj["objectType"])] = idx
                idx += 1
            # handle place object action space
            if obj["receptacle"] and ("PutObject",obj["objectType"]) not in action_to_idx:
                action_to_idx[("PutObject",obj["objectType"])] = idx
                idx += 1

        for i in range(len(self.Navigation_action)):
            action_to_idx[(self.Navigation_action[i])] = idx
            idx += 1

        self.IDX_TO_ACTION = {v:k for k,v in action_to_idx.items()}

    def step(self, action):
        '''This function defines how to convert action to what's needed for Thor'''
        # convert action from integer to string e.g:9 -> PickupObject
        self.action = action
        action_tuple= self.IDX_TO_ACTION[self.action]

        # Handle interactive action, like pickup, release, MoveArm
        if len(action_tuple) == 2:
            action_str, target = action_tuple
            for obj in self.controller.last_event.metadata["objects"]:
                if obj["objectType"] == target:
                    event = self.controller.step(action=action_str, objectId=obj["objectId"])
                    if event.metadata["lastActionSuccess"]:
                        break
        # Handle Navigation action
        else:
            self.controller.step(action=action_tuple)

class Action(enum.Enum):

    MoveAhead = 0     # since enum.auto() default start from index 1, not 0
    MoveBack = enum.auto()
    MoveLeft = enum.auto()
    MoveRight = enum.auto()
    LookUp = enum.auto()
    LookDown = enum.auto()
    RotateRight = enum.auto()
    RotateLeft = enum.auto()
    DropHandObject = enum.auto()
    OpenObject = enum.auto()
    PickupObject = enum.auto()
    PutObject = enum.auto()
    ThrowObject = enum.auto()
    ReleaseObject = enum.auto()

class Object(enum.Enum):

    Faucet = 0
    CellPhone = enum.auto()
    Apple = enum.auto()
    Orange = enum.auto()
    Knife = enum.auto()
    Bottle = enum.auto()
    Bread = enum.auto()
    Fork = enum.auto()
    Potato = enum.auto()
    Tomato = enum.auto()
    SoapBottle = enum.auto()
    Egg = enum.auto()
    CreditCard =enum.auto()
    WineBottle = enum.auto()
    PaperTowelRoll = enum.auto()
    Cup = enum.auto()
    PepperShake = enum.auto()
    Lettuce = enum.auto()
    ButterKnife = enum.auto()
    DishSponge = enum.auto()
    Spoon = enum.auto()
    Mug = enum.auto()
    Fridge = enum.auto()
    Drawe = enum.auto()
    Kettle = enum.auto()
    Book = enum.auto()
    StoveBurner = enum.auto()
    Drawer = enum.auto()
    CounterTop = enum.auto()
    Cabinet = enum.auto()
    Window = enum.auto()
    Sink = enum.auto()
    Floor = enum.auto()
    Microwave = enum.auto()
    Shelf = enum.auto()
    HousePlant = enum.auto()
    Toaster = enum.auto()
    CoffeeMachine = enum.auto()
    Pan = enum.auto()
    Plate = enum.auto()
    Vase = enum.auto()
    GarbageCan = enum.auto()
    Pot = enum.auto()
    Spatula = enum.auto()
    Bowl = enum.auto()
    SinkBasin = enum.auto()
    SaltShaker = enum.auto()
    PepperShaker = enum.auto()
    LightSwitch = enum.auto()
    ShelvingUnit = enum.auto()
    Statue = enum.auto()
    Stool = enum.auto()
    StoveKnob = enum.auto()
    Chair = enum.auto()
    Pen = enum.auto()
    Ladle = enum.auto()
    SprayBottle = enum.auto()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep_obj", default=None, type=list,
                        help="A list of specific types of objects want to keep，Control environment difficulty")
    parser.add_argument("--random_obj", default=False, type=bool,
                        help="True if you want to random initialize object position")
    parser.add_argument("--random_agent", default=False, type=bool,
                        help="True if you want to random initialize agent position")
    parser.add_argument("--dowesample", default=False, type=bool,
                        help="True if you want to dowsample pixel of image")

    args = parser.parse_args()
