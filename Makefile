cuda?=3
actors?=4
gpus?=1
cpus?=4
skip?=1
group?=''
notes?=''
agent?=muzero
search?=''
nojit?=0

debug?=0
debug_sup?=0
debug_sync?=1

wandb?=1
wandb_sync?=0
babyai_supervised_project?=babyai_supervised
babyai_online_project?=babyai_online
babyai_offline_project?=babyai_offline

folder?=../results/factored_muzero
wandb_dir?=../results/factored_muzero/wandb

wandb_entity?=wcarvalho92
task?=place_split_easy
size?=large

export PYTHONPATH:=$(PYTHONPATH):.

export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):/usr/local/cuda/targets/x86_64-linux/lib/:/usr/lib/x86_64-linux-gnu/:$(HOME)/miniconda3/envs/omodel/lib/:${HOME}/miniconda3/lib/

export JAX_DISABLE_JIT=$(nojit)

jupyter_lab:
	DISPLAY=3 \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	jupyter lab --port 4442 --no-browser --ip 0.0.0.0

babyai_dataset:
	python -m ipdb -c continue experiments/babyai_collect_data.py \
	--tasks_file=$(task) \
	--size=$(size) \
	--debug=$(debug)

babyai_datasets:
	python make_babyai_datasets.py \
	--debug=$(debug)

supervised_babyai_debug:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/babyai_supervised_trainer.py \
		--train_single=True \
		--make_path=True \
		--auto_name_wandb=True \
		--debug=$(debug_sup) \
		--folder="$(folder)/$(babyai_supervised_project)_debug" \
		--agent=$(agent) \
		--use_wandb=$(wandb_sync) \
		--wandb_project="$(babyai_supervised_project)_debug" \
		--wandb_entity=$(wandb_entity) \
		--wandb_group=$(group) \
		--wandb_notes=$(notes) \
		--wandb_dir=$(wandb_dir) \
		--search=$(search) \
		--agent=$(agent) \
		--tasks_file=$(task) \
		--size=$(size) \

offline_babyai_sync:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/babyai_offline_trainer.py \
		--train_single=True \
		--make_path=True \
		--auto_name_wandb=True \
		--make_dataset=True \
		--run_distributed=False \
		--debug=$(debug_sync) \
		--folder="$(folder)/$(babyai_offline_project)_sync" \
		--agent=$(agent) \
		--use_wandb=$(wandb_sync) \
		--wandb_project="$(babyai_offline_project)_sync" \
		--wandb_entity=$(wandb_entity) \
		--wandb_group=$(group) \
		--wandb_notes=$(notes) \
		--wandb_dir=$(wandb_dir) \
		--skip=$(skip) \
		--search=$(search) \
		--agent=$(agent) \
		--tasks_file=$(task) \
		--size=$(size) \
		--num_actors=$(actors) \
		--num_gpus=$(gpus) \
		--num_cpus=$(cpus)


online_babyai_sync:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/babyai_online_trainer.py \
		--train_single=True \
		--make_path=True \
		--auto_name_wandb=True \
		--run_distributed=False \
		--debug=$(debug_sync) \
		--folder="$(folder)/$(babyai_online_project)_sync" \
		--tasks_file=$(task) \
		--use_wandb=$(wandb_sync) \
		--wandb_project="$(babyai_online_project)_sync" \
		--wandb_entity=$(wandb_entity) \
		--wandb_group=$(group) \
		--wandb_notes=$(notes) \
		--wandb_dir=$(wandb_dir) \
		--skip=$(skip) \
		--search=$(search) \
		--num_gpus=$(gpus) \
		--num_cpus=2



supervised_babyai_run:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python experiments/babyai_supervised_trainer.py \
		--train_single=False \
		--debug=$(debug) \
		--folder="$(folder)/$(babyai_supervised_project)" \
		--use_wandb=$(wandb) \
		--wandb_project="$(babyai_supervised_project)" \
		--wandb_entity=$(wandb_entity) \
		--wandb_group=$(group) \
		--wandb_notes=$(notes) \
		--wandb_dir=$(wandb_dir) \
		--skip=$(skip) \
		--search=$(search) \
		--tasks_file=$(task) \
		--size=$(size) \
		--num_gpus=$(gpus) \
		--num_cpus=$(cpus)


offline_babyai_async:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python experiments/babyai_offline_trainer.py \
		--run_distributed=True \
		--folder="$(folder)/$(babyai_offline_project)_async" \
		--tasks_file=$(task) \
		--size=$(size) \
		--use_wandb=$(wandb) \
		--wandb_project="$(babyai_offline_project)_async" \
		--wandb_entity=$(wandb_entity) \
		--wandb_group=$(group) \
		--wandb_notes=$(notes) \
		--wandb_dir=$(wandb_dir) \
		--skip=$(skip) \
		--debug=$(debug) \
		--search=$(search) \
		--num_actors=$(actors) \
		--num_gpus=$(gpus) \
		--num_cpus=2


online_babyai_async:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python experiments/babyai_online_trainer.py \
		--run_distributed=True \
		--folder="$(folder)/$(babyai_online_project)_async" \
		--use_wandb=$(wandb) \
		--wandb_project="$(babyai_online_project)_async" \
		--wandb_entity=$(wandb_entity) \
		--wandb_group=$(group) \
		--wandb_notes=$(notes) \
		--wandb_dir=$(wandb_dir) \
		--skip=$(skip) \
		--debug=$(debug) \
		--search=$(search) \
		--num_actors=$(actors) \
		--num_gpus=$(gpus) \
		--num_cpus=$(cpus)