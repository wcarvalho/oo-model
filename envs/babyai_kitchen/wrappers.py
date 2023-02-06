import gym
import numpy as np

class MissionIntegerWrapper(gym.core.ObservationWrapper):
  """
  Wrapper to convert mission to integers.
  """

  def __init__(self, env, instr_preproc, max_length=30):
    super().__init__(env)

    self.instr_preproc = instr_preproc
    self.max_length = max_length

    self.observation_space.spaces['mission'] = gym.spaces.Box(
        low=0,
        high=1024,
        shape=(self.max_length, ),
        dtype='int32'
    )
  def observation(self, obs):
    obs_mission = np.zeros((self.max_length), dtype=np.int32)
    mission = self.instr_preproc(obs['mission'])
    assert len(mission) <= self.max_length
    obs_mission[:len(mission)] = mission

    obs['mission'] = obs_mission
    return obs

class RGBImgFullyObsWrapper(gym.core.ObservationWrapper):
  """
  Wrapper to use fully observable RGB image as the only observation output
  This can be used to have the agent to solve the gridworld in pixel space.
  It removes the direction key from the observation but keeps everything else.
  """

  def __init__(self, env, tile_size=8):
    super().__init__(env)

    self.tile_size = tile_size

    obs_shape = env.observation_space.spaces['image'].shape
    self.observation_space.spaces['image'] = gym.spaces.Box(
        low=0,
        high=255,
        shape=(self.env.height * tile_size,
               self.env.width * tile_size,
               3),
        dtype='uint8'
    )

  def observation(self, obs):
    env = self.unwrapped

    rgb_img = env.render(
        obs['image'],
        tile_size=self.tile_size,
        highlight=False
    )

    keys = obs.keys()
    new_obs = dict()
    for k in keys:
        if k == "direction": continue
        new_obs[k] = obs[k]
    new_obs['image'] = rgb_img
    return new_obs


class RGBImgPartialObsWrapper(gym.core.ObservationWrapper):
  """
  Wrapper to use partially observable RGB image as the only observation output
  This can be used to have the agent to solve the gridworld in pixel space.
  It removes the direction key from the observation but keeps everything else.
  """

  def __init__(self, env, tile_size=8):
    super().__init__(env)

    self.tile_size = tile_size

    obs_shape = env.observation_space.spaces['image'].shape
    self.observation_space.spaces['image'] = gym.spaces.Box(
        low=0,
        high=255,
        shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
        dtype='uint8'
    )

  def observation(self, obs):
    env = self.unwrapped

    rgb_img_partial = env.get_obs_render(
        obs['image'],
        tile_size=self.tile_size
    )

    keys = obs.keys()
    new_obs = dict()
    for k in keys:
        if k == "direction": continue
        new_obs[k] = obs[k]
    new_obs['image'] = rgb_img_partial
    return new_obs
