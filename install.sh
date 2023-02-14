conda create -n omodel python=3.9 -y

eval "$(conda shell.bash hook)"
conda activate omodel


conda env update --name omodel --file config.yaml

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
ACME
#############################################
git clone https://github.com/deepmind/acme.git _acme
git checkout 4525ade7015c46f33556e18d76a8d542b916f264
cd _acme
pip install --editable ".[jax,testing]"
cd ..

pip install "gym[atari]==0.23.0"