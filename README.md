
# Experiments

## Offline RL experiments
Not finished

## Online RL experiments

Running experiments:
```
online_sync  agent=muzero   # debugging 
online_async agent=muzero   # distributed setup for actor/evaluator/learner
online_many  search=$(name) # run many distributed experiments
```

## Directory structure
```
results/
  {experiment}/
    sync  # debugging sync
    named
```