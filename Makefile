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

wandb?=1
wandb_sync?=0
babyai_online_project?=babyai_online_muzero
babyai_offline_project?=babyai_offline_muzero
babyai_online_folder?=../results/factored_muzero/babyai_online
babyai_offline_folder?=../results/factored_muzero/babyai_offline

wandb_entity?=wcarvalho92

export PYTHONPATH:=$(PYTHONPATH):.

export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):/usr/local/cuda/targets/x86_64-linux/lib/:/usr/lib/x86_64-linux-gnu/:$(HOME)/miniconda3/envs/omodel/lib/:${HOME}/miniconda3/lib/

export JAX_DISABLE_JIT=$(nojit)

jupyter_lab:
	DISPLAY=3 \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	jupyter lab --port 5558 --no-browser --ip 0.0.0.0

babyai_datasets:
	python make_babyai_datasets.py \
	--debug=$(debug)

online_babyai_sync:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/babyai_online_trainer.py \
		--train_single=True \
		--run_distributed=False \
		--debug=1 \
		--folder="$(babyai_online_folder)_sync" \
		--agent=$(agent) \
		--use_wandb=$(wandb_sync) \
		--wandb_project=$(babyai_online_project) \
		--wandb_entity=$(wandb_entity) \
		--wandb_group=$(group) \
		--wandb_notes=$(notes) \
		--skip=$(skip) \
		--search=$(search) \
		--agent=$(agent) \
		--num_actors=$(actors) \
		--num_gpus=$(gpus) \
		--num_cpus=$(cpus)

offline_babyai_sync:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/babyai_offline_trainer.py \
		--train_single=True \
		--run_distributed=False \
		--debug=1 \
		--folder="$(babyai_offline_folder)_sync" \
		--agent=$(agent) \
		--use_wandb=$(wandb_sync) \
		--wandb_project=$(babyai_offline_project) \
		--wandb_entity=$(wandb_entity) \
		--wandb_group=$(group) \
		--wandb_notes=$(notes) \
		--skip=$(skip) \
		--search=$(search) \
		--agent=$(agent) \
		--num_actors=$(actors) \
		--num_gpus=$(gpus) \
		--num_cpus=$(cpus)


online_babyai_async:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/babyai_online_trainer.py \
		--run_distributed=True \
		--folder="$(babyai_online_folder)_async" \
		--agent=$(agent) \
		--use_wandb=$(wandb) \
		--wandb_project=$(babyai_online_project) \
		--wandb_entity=$(wandb_entity) \
		--wandb_group=$(group) \
		--wandb_notes=$(notes) \
		--skip=$(skip) \
		--debug=$(debug) \
		--search=$(search) \
		--num_actors=$(actors) \
		--num_gpus=$(gpus) \
		--num_cpus=$(cpus)

offline_babyai_async:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/babyai_offline_trainer.py \
		--run_distributed=True \
		--folder="$(babyai_offline_folder)_async" \
		--agent=$(agent) \
		--use_wandb=$(wandb) \
		--wandb_project=$(babyai_offline_project) \
		--wandb_entity=$(wandb_entity) \
		--wandb_group=$(group) \
		--wandb_notes=$(notes) \
		--skip=$(skip) \
		--debug=$(debug) \
		--search=$(search) \
		--num_actors=$(actors) \
		--num_gpus=$(gpus) \
		--num_cpus=$(cpus)