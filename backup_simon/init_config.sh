# required when importing Keras 2.0.9 with Theano backend
export MKL_THREADING_LAYER=GNU

# required for using GPU
export CUDA_HOME=/usr/local/cuda-8.0/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
PATH=${CUDA_HOME}/bin:${PATH}
export PATH
export LD_LIBRARY_PATH=/services/scratch/perception/cuda/lib64/:$LD_LIBRARY_PATH
export CPATH=/services/scratch/perception/cuda/include:$CPATH
export LIBRARY_PATH=/services/scratch/perception/cuda/lib64/:$LD_LIBRARY_PATH

# added by Miniconda3 installer
export PATH="/home/sileglai/miniconda3/bin:$PATH"
