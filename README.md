Dirichlet Diffusion Score Model 
==============

This repo contains the official implementation for the paper [Dirichlet diffusion score model for biological sequence generation](https://arxiv.org/abs/2305.10699). 

**Dirichlet Diffusion Score Model (DDSM)** is a continuous-time diffusion framework designed specificaly for modeling discrete data such as biological
sequences. We introduce a diffusion process defined in probability simplex space with stationary distribution being the Dirichlet distribution. This makes diffusion in continuous space natural for modeling discrete data. DDSM is the first approach for discrete data modeling with continuous-time  stochastic differential equation (SDE) diffusion in probability simplex space.

The Jax version of the code will be published soon. 

Installation instructions
---------- 
Please create a new conda or pip environment specifically for running DDSM.  

DDSM requires Python packages PyTorch (>=1.0). You can follow PyTorch installation steps [here](https://pytorch.org/get-started/locally/). 

If you plan to run promoter designer model, DDSM requires Selene (>=0.5.0). You can follow Selene installation steps [here](https://github.com/FunctionLab/selene).

Input data for sudoku and promoter designer experiment as well as model weights with the best performance can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.7943307) 

Tutorial
---------- 
An example notebook containing code for applying a toy model to binarized MNIST dataset is [here](https://github.com/jzhoulab/ddsm/blob/main/toy_example_bin_mnist.ipynb).

[Usage.md](USAGE.md) contains detailed information how to use other scripts provided in the repository. 

Benchmarks
----------
The evaluation is based on comparing generated sequences and human genome promoter sequences (ground truth) on the test chromosomes.
The metric SP-MSE is the MSE between the predicted promoter activity of generated sequences and human genome sequences (lower is better). Our model trained with DDSM outperforms models trained with other approaches:

| Model                                 | SP-MSE $\downarrow$ |
|---------------------------------------|---------------------|
| **DDSM (time dilation 4x)**           | **0.0334**          |
| DDSM (time dilation 2x)               | 0.0348              |
| DDSM (time dilation 1x)               | 0.0363              |
| D3PM-uniform / Multinomial Diffusion  | 0.0375              |
| Bit Diffusion (one-hot encoding)      | 0.0395              |
| Bit Diffusion (bit-encoding)          | 0.0414              | 

One can find more benchmarks on various datasets in the paper (see Publications)


License
-------
DDSM is distributed under a BSD-3-Clause license. See the [LICENSE file](LICENSE) for details. 

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
Pavel Avdeyev, Chenlai Shi, Yuhao Tan, Kseniia Dudnyk and Jian Zhou. "Dirichlet diffusion score model for biological sequence generation".

How to get help
---------------
The preferred way of asking questions about DDSM is the [discussions tab](https://github.com/jzhoulab/ddsm/discussions). Before posting an question, consider to look through the existing threads - it is possible that your question has already been answered. For report any bugs, please use the issues tracker.

In case you prefer personal communication, please contact Pavel at Pavel.Avdeev(at)UTSouthwestern.edu or  Jian at Jian.Zhou(at)UTSouthwestern.edu.
