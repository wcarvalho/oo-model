import os.path
import json
from envs.babyai_kitchen.world import Kitchen
import re
import numpy as np


class InstructionsPreprocessor(object):
  def __init__(self, path):
    if os.path.exists(path):
        self.vocab = json.load(open(path))
    else:
        raise FileNotFoundError(f'No vocab at "{path}"')

  def __call__(self, mission, device=None):
    """Copied from BabyAI
    """
    tokens = re.findall("([a-z]+)", mission.lower())
    return np.array([self.vocab[token] for token in tokens])

