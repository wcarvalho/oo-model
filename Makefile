cuda?=3
actors?=4
gpus?=1
cpus?=4
debug?=0
skip?=1
data_file?='data/place.debug.pkl'
group?=''
notes?=''
wandb_test?=0
wandb?=1
agent?=muzero
search?=''
nojit?=0
task?=pickup

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

offline:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/offline_rl.py \
	--run_distributed=False \
	--debug=$(debug) \

offline_async:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/offline_rl.py

online_sync:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/online_rl.py \
		--agent=$(agent) \
		--use_wandb=$(wandb_test) \
		--wandb_project=muzero_debug \
		--wandb_entity=wcarvalho92 \
		--wandb_group=$(group) \
		--debug=$(debug) \
		--tasks_file=$(task) \
		--wandb_notes=$(notes)

online_async:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/online_rl.py \
		--agent=$(agent) \
		--use_wandb=$(wandb_test) \
		--wandb_project=muzero_debug \
		--wandb_entity=wcarvalho92 \
		--wandb_group=$(group) \
		--wandb_notes=$(notes) \
		--run_distributed=True \
		--debug=$(debug)

online_many:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python experiments/train_many.py \
		--agent=$(agent) \
		--spaces="experiments.online_rl_searches" \
		--use_wandb=$(wandb) \
		--wandb_project=muzero \
		--wandb_entity=wcarvalho92 \
		--wandb_group=$(group) \
		--wandb_notes=$(notes) \
		--skip=$(skip) \
		--debug=$(debug) \
		--tasks_file=$(task) \
		--search=$(search) \
		--num_actors=$(actors) \
		--num_gpus=$(gpus) \
		--num_cpus=$(cpus)