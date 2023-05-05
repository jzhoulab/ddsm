Dirichlet Diffusion Score Model 
==============

This repo contains the official implementation for the paper [Dirichlet diffusion score model for biological sequence generation](doi_here). 

**Dirichlet Diffusion Score Model (DDSM)** is a continuous-time diffusion framework designed specificaly for modeling discrete data such as biological
sequences. We introduce a diffusion process defined in probability simplex space with stationary distribution being the Dirichlet distribution. This makes diffusion in continuous space natural for modeling discrete data. DDSM is the first approach for discrete data modeling with continuous-time  stochastic differential equation (SDE) diffusion in probability simplex space.

The Jax version of code will be published soon. 

Installation instructions
---------- 
Please create a new conda or pip environment specifically for running DDSM. DDSM requires Python packages PyTorch (>=1.0). You can follow PyTorch installation steps [here](https://pytorch.org/get-started/locally/). 

If you plan to run TSS model, DDSM requires Selene (>=0.5.0). You can follow Selene installation steps [here](https://github.com/FunctionLab/selene).

Usage
---------- 
TBA

Benchmarks
----------
TBA - biological sequences comparison

One can find more benchmarks on various datasets in our paper (doi here) 


License
-------
DDSM is distributed under a Apache License 2.0.  See the [LICENSE file](LICENSE) for details. 

Credits
-------
DDSM is developed in [Jian Zhou's lab at UTSW](https://zhoulab.io/).

* Pavel Avdeyev
* Chenlai Shi
* Yuhao Tan
* Kseniia Dudnyk
* Jian Zhou

Publications
------------
Pavel Avdeyev, Chenlai Shi, Yuhao Tan, Kseniia Dudnyk and Jian Zhou. "Dirichlet diffusion score model for biological sequence generation". (details here) 

How to get help
---------------
A preferred way report any problems or ask questions about DDSM is the [issue tracker](https://github.com/jzhoulab/ddsm/issues). Before posting an issue/question, consider to look through the existing issues (opened and closed) - it is possible that your question has already been answered.

In case you prefer personal communication, please contact Jian at (placeHolder)
