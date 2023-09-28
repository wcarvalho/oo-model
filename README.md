
# Experiments

To use wandb, set the following environment variables (they default to my own):
- `wandb_entity` (wandb login info)
- `wandb_project` (which project to log to)

NOTE: that `wandb_group` is overwritten by `$(search)`

## Supervised Learning
```
make supervised_babyai_debug  search=$(name)   # debugging 
make supervised_babyai_run    search=$(name)   # run a larger experiment
```

## Online
```
make online_babyai_sync agent=muzero     # debugging, synchronous actor/learner/eval
make online_babyai_async  search=$(name) # run multiple experiments, everything asynchronous
```
  
## Directory structure
```
results/
  {experiment}/
    sync  # debugging sync
    named
```


# Utilities

**batch git commits**: update `updates.yml` and run `python git-commit.py`. suggest copying both files.