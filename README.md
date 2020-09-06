# Dynamical Variational Autoencoders

This repository contains the code for this paper:

**Dynamical Variational Autoencoders: A Review of Recent Trend**  
[Laurent Girin](http://www.gipsa-lab.grenoble-inp.fr/~laurent.girin/cv_en.html), [Simon Leglaive](https://sleglaive.github.io/index.html), [Xiaoyu Bie](https://team.inria.fr/perception/team-members/xiaoyu-bie/), [Julien Diard](https://diard.wordpress.com/), [Thomas Hueber](http://www.gipsa-lab.grenoble-inp.fr/~thomas.hueber/), [Xavier Alameda-Pineda](http://xavirema.eu/)  
**[[Paper](https://hal.inria.fr/hal-02926215)]**

More precisely, it is a re-implementation of the following models in Pytorch for speech re-synthesis task:
- [VAE](https://arxiv.org/abs/1312.6114)
- [DKF](https://arxiv.org/abs/1609.09869)
- [KVAE](https://papers.nips.cc/paper/6951-a-disentangled-recognition-and-nonlinear-dynamics-model-for-unsupervised-learning)
- [STORN](https://arxiv.org/abs/1411.7610)
- [VRNN](https://arxiv.org/abs/1506.02216)
- [SRNN](https://arxiv.org/abs/1605.07571)
- [RVAE](https://arxiv.org/abs/1910.10942)
- [DSAE](https://arxiv.org/abs/1803.02991)


## Bibtex
If you find this code useful, please star the project and consider citing:

```
TODO
```


## Installation instructions

### Dependencies

This code has been tested on Ubuntu 16.04, Python 3.7, Pytorch=1.3.1, CUDA 9.2, NVIDIA Titan X and NVIDIA Titan RTX, it deepends on:

- soundfile
- librosa
- CUDA 9.2


## Usage

`dvae` has been designed to be designed to be easily used in a modular way. All you need to do is to specify a configure file path to innitiale a model, then you could either:

- train(): train your model with training/validation dataset specified in your configure file
- generate(): generate a reconstructed audio with an input audio file
- eval(): evaluate reconstrcuted audio file with specified metric (`rmse`, `pesq`, `stoi`, `all`) 
- test(): apply reconstruction for all audio files with indicated test dataset directory path



### Example

```python
# instantiate a model
from dvae import LearningAlgorithm
cfg_file = 'config.ini'
learning_algo = LearningAlgorithm(config_file=cfg_file)

# train your model
learning_algo.train()

# generate audio with model state in cache
# reconstructed audio will be saved in the same path as input audio file, named as 'audio_001_recon.wav'
audio_ref = 'audio_001.wav'
learning_algo.generate(audio_orig=audio_ref, audio_recon=None, state_dict_file=None)

# generate audio with given model state
# reconstructed audio will be saved in the given path
audio_recon = 'recon/audio_001_recon.wav'
model_state = 'model_state.pt'
learning_algo.generate(audio_orig=audio_ref, audio_recon=audio_recon, state_dict_file=model_state)

# evaluate audio quality with model state in cache
score_rmse = learning_algo.eval(audio_ref=audio_ref, audio_est=audio_recon, metric='rmse', state_dict_file=None) # only RMSE
score_rmse, score_pesq, score_stoi = learning_algo.eval(audio_ref=audio_ref, audio_est=audio_recon, metric='all', state_dict_file=None) # both RMSE, PESQ and STOI

# test model on test dataset with model state in cache
test_data_dir = 'data_to_test'
list_score_rmse, list_score_pesq, list_score_stoi = learning_algo.test(data_dir=test_data_dir, state_dict_file=None)
```



## Main results

To show training results and re-synthesis results here


## Tips

- One who want to use the package on `MacOS` should disable `speechmetrics` since there exist some compilation errors for `pypesq` on `MacOS`
- Python 3.8 support requires Tensorflow 2.2 or later ([link](https://www.tensorflow.org/install/pip)), so it is recommended to use Python 3.7 since `speechmetrics` package needs `Tensorflow 2.0`

### To add license

'''
Software XX
Copyright Inria [and XX if needed]
Year YYY
Contact : name.surname@inria.fr, ...

The software XX is provided "as is", for reproducibility purposes only.
The user is not authorized to distribute the software XX, modified or not.
The user expressly undertakes not to remove, or modify, in any manner, the intellectual property notice attached to the software XX.
'''
