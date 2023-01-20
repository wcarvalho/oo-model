cuda?=3
debug?=3

export PYTHONPATH:=$(PYTHONPATH):.
# export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(HOME)/miniconda3/envs/acmejax/lib/:${HOME}/miniconda3/lib

collect_data:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python experiments/collect_data.py \
	--testing $(debug)

offline:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python experiments/offline_model_learning.py