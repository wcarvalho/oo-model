conda create -n omodel2 python=3.9 -y

eval "$(conda shell.bash hook)"
conda activate omodel2


conda env update --name omodel2 --file env.yaml

# conda install -c anaconda -y cudnn
##############################################
# BabyAI
##############################################
git clone https://github.com/maximecb/gym-minigrid.git _gym-minigrid
cd _gym-minigrid
git checkout 03cf21f61bce58ab13a1de450f6269edd636183a
cd ..
cp install/minigrid_setup.py _gym-minigrid/setup.py
cd _gym-minigrid
pip install --editable .
cd ..

git clone https://github.com/mila-iqia/babyai.git _babyai
cp install/babyai_setup.py _babyai/setup.py
cd _babyai
pip install --editable .
cd ..


#############################################
# ACME
#############################################
git clone https://github.com/deepmind/acme.git _acme
cd _acme
git checkout 4525ade7015c46f33556e18d76a8d542b916f264
pip install --editable ".[jax,testing,envs]"
cd ..

#############################################
# jax
# https://github.com/google/jax
#############################################
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/targets/x86_64-linux/lib/
pip install "jax[cuda11_cudnn82]==0.4.4" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install chex==0.1.6
pip install gym[accept-rom-license]

#############################################
# Setup activate/deactivate with correct PYTHONPATH and LD_LIBRARY_PATH
#############################################
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo 'export PYTHONPATH=$PYTHONPATH:.' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/lib/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/targets/x86_64-linux/lib/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'unset LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh


# test whether jax installed properly...
python -c "import jax; jax.random.split(jax.random.PRNGKey(42), 2); print('hello world')"