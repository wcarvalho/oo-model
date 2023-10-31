# Install
```
bash install.sh
```

# Setting up your own experiments

1. for online, copy `experiments/babyai_online_trainer.py` or for supervised, copy `experiments/babyai_supervised_trainer.py`
2. you'll set searches in the `sweep(search: str)` function

## Useful command line arguments to set:
- `folder`: directory to place experiments. recommend not putting in same directory
- `train_single`: if True, just runs first config option. If False, runs all in parallel
- `folder`: directory to place experiment results
- `wandb_entity` (wandb login info)
- `wandb_project` (which project to log to)
- `debug`: whether to use debug settings.
- `search`: which search to run
- `num_gpus`: how to gpus to use for run. defaults to `1`. If you set `.5`, will do 2 runs for each gpu.

set `CUDA_VISIBLE_DEVICES` to which GPU to use. e.g. `CUDA_VISIBLE_DEVICES=0` uses the 0-th GPU.

### Running experiments (using online RL as example)
#### Debugging
replace `$(x)` with the argument. e.g. `$(search)` with muzero_test.
```
CUDA_VISIBLE_DEVICES=$(cuda) \
python experiments/babyai_online_trainer.py \  # change this
  --train_single=True \
  --folder=$(folder) \
  --wandb_project=$(wandb_project) \
  --wandb_entity=$(wandb_entity) \
  --search=$(search) \
  --num_gpus=$(gpus)
```
**example**

```
CUDA_VISIBLE_DEVICES=0 \
python experiments/babyai_online_trainer.py \
  --train_single=True \
  --folder="/tmp/results" \
  --use_wandb=False \
  --search="default" \
  --num_gpus=1
```

#### Running multiple in parallel
You'll want to set `--train_single=False` and use more GPUs, e.g.
**recommendation**: either use different `wand_project` for parallel/debugging (to not pollute wandb with debugging runs) or set `--use_wandb=False` for debugging.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python experiments/babyai_online_trainer.py \  # change this
  --train_single=False \
  --folder=$(folder) \
  --wandb_project=$(wandb_project) \
  --wandb_entity=$(wandb_entity) \
  --search=$(search) \
  --num_gpus=$(gpus)
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