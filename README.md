# Variational Autoencoders in Pytorch

`VAEs.py`: Contains the classes `VAE` and `RVAE` implementing respectively the FFNN-VAE and (B)RNN-VAE.

`speech_dataset.py`: Contains the dataset classes for generating frames or sequences of frames from a speech dataset (tested on TIMIT and WSJ0)

`train_*_WSJ0.py`: Scripts for training VAEs on WSJ0 dataset.

`test_.py`: Old scripts to test VAEs in an encoding-decoding fashion.

Shell scripts are for running experiments on the cluster.