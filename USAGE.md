DDSM MANUAL 
========== 

We provide scripts for training and evaluating models on three datasets: binary MNIST, sudoku, and promoter design. Before proceeding further, please donwload model weights and input data from [Zenodo repository](https://doi.org/10.5281/zenodo.7943307) to DDSM directory. For example, one can run the following commands:  

```
git clone https://github.com/jzhoulab/ddsm.git 
cd ddsm 
wget 'https://zenodo.org/record/7943307/files/best_models.tar.gz?download=1' -O best_models.tar.gz 
wget 'https://zenodo.org/record/7943307/files/data.tar.gz?download=1' -O data.tar.gz 
tar -xvfz data.tar.gz 
tar -xvfz best_models.tar.gz
```

After running commands above, one should have `best_models_weights` and `data` directory in the `ddsm` folder. Please, check README file in `data/promoter_design` folder. 

Binary Mnist
----------- 
We need to presample noise for our Dirichlet/Jacobi process. It can be done via `presample_noise.py` script. 

```
python presample_noise.py -n 100000 -c 2 -t 4000 --max_time 4 --out_path binary_mnist/
```

The file `steps4000.cat2.time4.0.samples100000.pth` should appear in `binary_mnist` folder after script finishes. The next steps are

```
cd binary_mnist
python eval_bin_mnist.py
```

Evaluation of the likelihood takes long time. At the end, one should be able to reproduce results from DDSM paper provided in the Table 1 (one needs multiply likelihood from the output by 28 * 28).  

Training of binary mnist model are similar to the code provided in [toy example](https://github.com/jzhoulab/ddsm/blob/main/toy_example_bin_mnist.ipynb). 

Sudoku
-------- 
As a first step, we need to presample noise. It can be done via `presample_noise.py` script. 

```
python presample_noise.py -n 100000 -c 9 -t 400 --max_time 1 --out_path sudoku/
```

The file `steps400.cat9.time1.0.samples100000.pth` should appear in `sudoku` folder.

The second step is to train model. The following code will train the sudoku model.  
```
cd sudoku 
python train_sudoku.py 
```
For reproducing results in the paper, we recommend training model for 1500-2000 epochs. 

For evaluating model, you can use the provided notebook `eval_sudoku.ipynb`. It reproduces results described in the 
Table 2 of the DDSM paper. We supplied our best model in `best_model_weights` directory. 


Promoter Design
-------- 

You need to generate `.mmap` file for human reference first. The script `data/promoter_design/make_genome_memmap.py` should help with that. 

As usual, we need to presample noise. It can be done via the following command: 
```
python presample_noise.py -n 100000 -c 4 -t 400 --max_time 4 ---speed_balance --out_path promoter_design/
```

The file `steps400.cat4.speed_balance.time4.0.samples100000.pth` should appear in `promoter_design` folder.

For training promoter design model, one can use the commands below. For running code, you need to have 
[Selene](https://github.com/FunctionLab/selene) installed in your environment. The `external` folder must be also 
reachable from `promoter_design` directory. 
```
cd promoter_design
python train_promoter_designer.py
```

We also provide `eval_promoter_designer.ipynb` notebook. Using this notebook, you can recreate pictures provided 
in the paper. 
