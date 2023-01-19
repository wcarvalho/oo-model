import json
from tqdm import trange
from pprint import pprint
import babyai.utils

from envs.babyai_kitchen.levelgen import KitchenLevel
from envs.babyai_kitchen.tasks import TASKS

def main():
  # ======================================================
  # create object to store vocabulary
  # ======================================================
  vocab = babyai.utils.format.Vocabulary(
    model_name='kitchen',
    )


  # ======================================================
  # load env
  # ======================================================
  env = KitchenLevel(
      task_kinds=list(TASKS.keys()),
      use_time_limit=False)

  for task, Cls in TASKS.items():
    instance = Cls(env.kitchen)
    mission = instance.abstract_rep
    [vocab[x] for x in mission.split(" ")]

  for object in env.kitchen.objects:
    vocab[object.type]


  pprint(vocab.vocab)
  vocab.save(path="data/babyai_kitchen/vocab.json")



if __name__ == "__main__":
    main()
