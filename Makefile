cuda?=3
actors?=4
gpus?=1
cpus?=4
debug?=0
skip?=1
group?=''
notes?=''
wandb?=1
agent?=muzero
search?=''
nojit?=0

babyai_online_project?=babyai_online_muzero
babyai_offline_project?=babyai_offline_muzero
wandb_entity?=wcarvalho92

export PYTHONPATH:=$(PYTHONPATH):.

export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):/usr/local/cuda/targets/x86_64-linux/lib/:/usr/lib/x86_64-linux-gnu/:$(HOME)/miniconda3/envs/omodel/lib/:${HOME}/miniconda3/lib/

export JAX_DISABLE_JIT=$(nojit)

jupyter_lab:
	DISPLAY=3 \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	jupyter lab --port 5558 --no-browser --ip 0.0.0.0

collect_data:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/collect_data.py \
	--debug=$(debug) \

online_babyai_sync:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python experiments/babyai_online_trainer.py \
		--agent=$(agent) \
		--use_wandb=$(wandb) \
		--wandb_project=$(babyai_project) \
		--wandb_entity=$(wandb_entity) \
		--wandb_group=$(group) \
		--wandb_notes=$(notes) \
		--skip=$(skip) \
		--debug=$(debug) \
		--search=$(search) \
		--agent=$(agent) \
		--num_actors=$(actors) \
		--num_gpus=$(gpus) \
		--num_cpus=$(cpus)

offline_babyai_sync:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python experiments/babyai_offline_trainer.py \
		--agent=$(agent) \
		--use_wandb=$(wandb) \
		--wandb_project=$(babyai_offline_project) \
		--wandb_entity=$(wandb_entity) \
		--wandb_group=$(group) \
		--wandb_notes=$(notes) \
		--skip=$(skip) \
		--debug=$(debug) \
		--search=$(search) \
		--agent=$(agent) \
		--num_actors=$(actors) \
		--num_gpus=$(gpus) \
		--num_cpus=$(cpus)


online_babyai_async:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python experiments/babyai_online_trainer.py \
		--run_distributed=True \
		--agent=$(agent) \
		--use_wandb=$(wandb) \
		--wandb_project=$(babyai_project) \
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
	python experiments/babyai_offline_trainer.py \
		--run_distributed=True \
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