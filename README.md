## Survival Kernets: Scalable and Interpretable Deep Kernel Survival Analysis with an Accuracy Guarantee

Author: George H. Chen (georgechen [at symbol] cmu.edu)

This code accompanies the paper:

> George H. Chen. "Survival Kernets: Scalable and Interpretable Deep Kernel Survival Analysis with an Accuracy Guarantee".\
> \[[arXiv](https://arxiv.org/abs/2206.10477)\]

### Code requirements

- Anaconda Python 3 (tested with Anaconda version 2020.07 running Python 3.8.3)
- Additional packages: hnswlib, lifelines, xgboost, pytorch (tested with PyTorch version 1.7.1 with CUDA 11.0); for more precise details, see the [requirements.txt](requirements.txt) file (includes the packages we just mentioned along with various other packages that should already come with Anaconda Python 3) which can be installed via `pip install -r requirements.txt`
- cython compilation is required for the random survival forests implementation used (in this repository, random survival forests are only used by the DKSA baseline):

```
python setup_random_survival_forest_cython.py build_ext --inplace
```

Note that this repository comes with slightly modified versions of Haavard Kvamme's [PyCox](https://github.com/havakv/pycox) and [torchtuples](https://github.com/havakv/torchtuples) packages (some bug fixes/print flushing). Also, some files are from my earlier [DKSA repository](https://github.com/georgehc/dksa/) (`neural_kernel_survival.py`, `npsurvival_models.py`, `random_survival_forest_cython.pyx`, `setup_random_survival_forest_cython.py`). Kvamme's code is under a BSD 2-clause license whereas my code is under an MIT license.

The experiments in the paper were run on identical Ubuntu 20.04.2 LTS instances, each with an Intel Core i9-10900K CPU @ 3.70GHz (10 cores, 20 threads), 64GB RAM, and an Nvidia Quadro RTX 4000 (8GB GPU RAM).

The main code for survival kernets with TUNA is in `demo_tuna_kernet.py` and without TUNA is in `demo_kernet.py`.

In the paper, I tested survival kernets on four datasets (ROTTERDAM/GBSG, SUPPORT, UNOS, and KKBOX). However, for the public release of the code, support for UNOS has been removed for simplicity as the UNOS dataset requires special access. As for the other datasets:

- The ROTTERDAM/GBSG (Foekens et al 2000; Schumacher et al 1994) datasets (train on Rotterdam and test on GBSG) is taken from the [DeepSurv (Katzman et al 2018) github repo](https://github.com/jaredleekatzman/DeepSurv)
- The SUPPORT (Knaus et al 1995) dataset was obtained from [https://hbiostat.org/data/](https://hbiostat.org/data/)
- The KKBOX dataset needs to be downloaded first using the PyCox code (check PyCox documentation for how to do this; at the time of writing, getting KKBOX with PyCox requires the Kaggle API to be installed first, including the user to have Kaggle credentials). The raw version of the KKBOX dataset (that is not preprocessed by PyCox) is available from [https://www.kaggle.com/c/kkbox-churn-prediction-challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge)

### Training different models and comparing them in terms of time-dependent concordance index

*Important:* The code currently is written to report **negative** time-dependent concordance indices, so expect to see validation and test set losses that are negative values between -1 and 0 (although occasionally the loss might be reported as infinity if there was some numerical issue encountered).

As an example of how to use the code, from within the base directory, to train the baseline XGBoost model on the SUPPORT dataset, you can run in a terminal:

```
python demo_xgb.py cfg_baselines.ini
```

To train one of the other baseline models, you can simply replace `demo_xgb.py` in the above terminal command with one of the other baseline model demo scripts (`demo_enet_cox.py`, `demo_deepsurv.py`, `demo_deephit.py`, `demo_dcm.py`, or `demo_dksa.py`). Hyperparameter grids and other settings are in `cfg_baselines.ini`. By default, output files for baseline models are stored in `./out_baselines/`, where the main test set metrics by default get stored to `./out_baselines/*_test_metrics*.csv` (includes test set time-dependent concordance indices). An example of what the output directory looks like when we run the above command for training XGBoost is available in `./example_out_baselines/`.

Next, to train survival kernets with TUNA, an XGBoost model must be trained first. For instance, if you already ran the line above (`python demo_xgb.py cfg_baselines.ini`) and the XGBoost model was successfully trained, then running the command below will train a survival kernet model with no sample splitting, TUNA warm-start, and summary fine-tuning (i.e., the TUNA-KERNET (NO SPLIT, SFT) model in the paper):

```
python demo_tuna_kernet.py cfg_kernet_nosplit_sft.ini
```

The hyperparameter grid and other settings are in `cfg_kernet_nosplit_sft.ini`. By default, output files for survival kernet variants with no sample splitting (no sample splitting means that pre-training and training sets are the same) get stored in `./out_kernets_nosplit/` whereas output files for survival kernet variants with sample splitting get stored in `./out_kernets_split/` (just as with the baselines, within the output directory, there will be test metrics stored in a `*_test_metrics*.csv` file after the model has been evaluated on the test set). An example of what the output directory looks like when we run the above command for training a TUNA-KERNET (NO SPLIT, SFT) model is available in `./example_out_kernets_nosplit/`. For a little more detail on how to train the different survival kernet variants, read the comments in `demo.sh`.

Note that the configuration .ini files are currently written to only use the SUPPORT dataset (specifically there is a line that says `datasets = ['support']`). If you want to run experiments on both the SUPPORT dataset and the ROTTERDAM/GBSG dataset, then you would set `datasets = ['rotterdam-gbsg', 'support']` in the .ini files. If you have successfully downloaded the KKBOX dataset through the PyCox interface, then you can specify that you also want to train and evaluate on the KKBOX dataset (e.g., to run experiments on ROTTERDAM/GBSG, SUPPORT, and KKBOX, you would set `datasets = ['rotterdam-gbsg', 'support', 'kkbox']` in the .ini files). The actual file that handles loading of the datasets is `datasets.py`, which can be modified to support additional datasets.

### Example of how to do cluster visualization with a survival kernet model

We have included a Jupyter notebook that shows how to make the visualizations for the SUPPORT dataset for survival kernets (specifically the TUNA-KERNET (NO SPLIT, SFT) model although in the Jupyter notebook we also illustrate how to get survival curves using the Kaplan-Meier estimator instead, which would correspond to the TUNA-KERNET (NO SPLIT) model); see `example_cluster_visualization.ipynb`. Note that it is possible to modify this visualization code to also work with the KKBOX dataset.
