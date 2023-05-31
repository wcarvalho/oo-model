import pickle 
from absl import logging
from pprint import pprint
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

def update_config(config, strict: bool = True, **kwargs):
  for k, v in kwargs.items():
    if not hasattr(config, k):
      message = f"Attempting to set unknown attribute '{k}'"
      if strict:
        raise RuntimeError(message)
      else:
        logging.warning(message)
    setattr(config, k, v)
