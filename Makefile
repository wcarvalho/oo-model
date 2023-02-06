cuda?=3
debug?=3
data_file?='data/place.debug.pkl'

export PYTHONPATH:=$(PYTHONPATH):.
export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(HOME)/miniconda3/envs/omodel/lib/:${HOME}/miniconda3/lib

collect_data:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/collect_rlds_data.py \
	--debug=$(debug) \

offline:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/offline_model_learning.py \
	--run_distributed=False \
	--debug=$(debug) \

offline_async:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/offline_model_learning.py


online:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/train.py
