mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo 'export export PYTHONPATH:=$PYTHONPATH:.' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/targets/x86_64-linux/lib/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'unset LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh