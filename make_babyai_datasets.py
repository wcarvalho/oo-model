from absl import app
from absl import flags
import itertools
import subprocess

from experiments import experiment_builders

FLAGS = flags.FLAGS

def make_command(tasks_file, room_size, partial_obs, debug=False):
  command = f"""python experiments/babyai_collect_data.py
    --tasks_file={tasks_file}
    --room_size={room_size}
    --partial_obs={partial_obs}
    --debug={int(debug)}
    """
  print(command)
  command = command.replace("\n", '')
  return command

def main(unused_argv):
  tasks_files = ['place_split_easy', 'place_split_medium', 'place_split_hard']
  room_sizes = [7]
  partial_obs = [False, True]

  debug = FLAGS.debug
  for (t, r, p) in itertools.product(tasks_files, room_sizes, partial_obs):
    command = make_command(tasks_file=t, room_size=r, partial_obs=p, debug=debug)
    process = subprocess.Popen(command, shell=True)
    if debug:
      process.wait()
      break

if __name__ == '__main__':
  app.run(main)
