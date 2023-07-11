#!/usr/bin/env python
"""
This runs the DKSA baseline, specifically the version that fits a random
survival forest (RSF) model first. This code uses the RSF and DKSA
implementations from the original DKSA repository:

    https://github.com/georgehc/dksa

Note that the code here is just to get the existing DKSA code to work with the
new experimental harness; no effort has been made to further optimize the
existing DKSA code.

Author: George H. Chen (georgechen [at symbol] cmu.edu)
"""
import ast
import configparser
import csv
import gc
import hashlib
import numpy as np
import os
import pandas as pd
import pickle
import random
import shutil
import sys
import time
import uuid
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# the imports could be affected by the CUBLAS var above
import torch
import torch.nn as nn
# torch.set_deterministic(True)  # causes problems with survival analysis
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torchtuples as tt
from npsurvival_models import RandomSurvivalForest
from datasets import load_dataset
from metrics import neg_cindex_td
from models import Hypersphere
from neural_kernel_survival import NKS, NKSDiscrete
from pycox.preprocessing.label_transforms import LabTransDiscreteTime


estimator_name = 'rsf'
finetune_estimator_name = 'dksa'

if not (len(sys.argv) == 2 and os.path.isfile(sys.argv[1])):
    print('Usage: python "%s" [config file]' % sys.argv[0])
    sys.exit(1)

config = configparser.ConfigParser()
config.read(sys.argv[1])

n_experiment_repeats = int(config['DEFAULT']['n_experiment_repeats'])
fix_test_shuffle_train = \
    int(config['DEFAULT']['fix_test_shuffle_train']) > 0
val_ratio = float(config['DEFAULT']['simple_data_splitting_val_ratio'])
datasets = ast.literal_eval(config['DEFAULT']['datasets'])
output_dir = config['DEFAULT']['output_dir']
method_header = 'method: %s' % estimator_name
method_random_seed = int(config[method_header]['random_seed'])
patience = int(config[method_header]['early_stopping_patience'])
max_n_cores = int(config['DEFAULT']['max_n_cores'])
verbose = int(config['DEFAULT']['verbose']) > 0

compute_bootstrap_CI = int(config['DEFAULT']['compute_bootstrap_CI']) > 0
bootstrap_CI_coverage = float(config['DEFAULT']['bootstrap_CI_coverage'])
bootstrap_n_samples = int(config['DEFAULT']['bootstrap_n_samples'])
bootstrap_random_seed = int(config['DEFAULT']['bootstrap_random_seed'])

mds_random_seed = int(config['DEFAULT']['mds_random_seed'])
mds_n_init = int(config['DEFAULT']['mds_n_init'])

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)

if max_n_cores <= 0:
    n_jobs = os.cpu_count()
else:
    n_jobs = min(max_n_cores, os.cpu_count())

pretrain_frac = float(config[method_header]['pretrain_frac'])
hyperparams = \
    [(max_features, min_samples_leaf)
     for max_features
     in ast.literal_eval(config['method: %s'
                                % estimator_name]['max_features'])
     for min_samples_leaf
     in ast.literal_eval(config['method: %s'
                                % estimator_name]['min_samples_leaf'])]

finetune_method_header = 'method: %s' % finetune_estimator_name
finetune_method_random_seed = \
    int(config[finetune_method_header]['random_seed'])
max_n_epochs = int(config[finetune_method_header]['max_n_epochs'])
lr_range = ast.literal_eval(config[finetune_method_header]['learning_rate'])
if 'warmstart_learning_rate' in config[finetune_method_header]:
    warmstart_lr_range = \
        ast.literal_eval(
            config[finetune_method_header]['warmstart_learning_rate'])
else:
    warmstart_lr_range = lr_range
warmstart_hyperparams = \
    [(squared_radius, n_layers, n_nodes, batch_size, max_n_epochs, lr)
     for squared_radius
     in ast.literal_eval(config[finetune_method_header]['squared_radius'])
     for n_layers
     in ast.literal_eval(config[finetune_method_header]['n_layers'])
     for n_nodes
     in ast.literal_eval(config[finetune_method_header]['n_nodes'])
     for batch_size
     in ast.literal_eval(config[finetune_method_header]['batch_size'])
     for lr
     in warmstart_lr_range]
n_durations_range = \
    ast.literal_eval(config[finetune_method_header]['n_durations'])
finetune_hyperparams = \
    [(pretrain_frac, n_durations, batch_size, max_n_epochs, lr)
     for n_durations
     in n_durations_range
     for batch_size
     in ast.literal_eval(config[finetune_method_header]['batch_size'])
     for lr
     in lr_range]

hyperparam_hash = hashlib.sha256()
hyperparam_hash.update(str(hyperparams).encode('utf-8'))
hyperparam_hash = hyperparam_hash.hexdigest()

warmstart_hyperparam_hash = hashlib.sha256()
warmstart_hyperparam_hash.update(str(hyperparams).encode('utf-8'))
warmstart_hyperparam_hash.update(str(warmstart_hyperparams).encode('utf-8'))
warmstart_hyperparam_hash = warmstart_hyperparam_hash.hexdigest()

finetune_hyperparam_hash = hashlib.sha256()
finetune_hyperparam_hash.update(str(hyperparams).encode('utf-8'))
finetune_hyperparam_hash.update(str(warmstart_hyperparams).encode('utf-8'))
finetune_hyperparam_hash.update(str(finetune_hyperparams).encode('utf-8'))
finetune_hyperparam_hash = finetune_hyperparam_hash.hexdigest()

validation_string = 'vr%f' % val_ratio

full_estimator_name = '%s_oldtuna=%s' % (finetune_estimator_name, estimator_name)

output_test_table_filename \
    = os.path.join(output_dir,
                   '%s_experiments%d_%s_test_metrics_%s.csv'
                   % (full_estimator_name,
                      n_experiment_repeats,
                      validation_string,
                      finetune_hyperparam_hash))
output_test_table_file = open(output_test_table_filename, 'w')
test_csv_writer = csv.writer(output_test_table_file)
if compute_bootstrap_CI:
    test_csv_writer.writerow(['dataset',
                              'experiment_idx',
                              'method',
                              'loss',
                              'loss_CI_lower',
                              'loss_CI_upper'])
else:
    test_csv_writer.writerow(['dataset',
                              'experiment_idx',
                              'method',
                              'loss'])


for experiment_idx in range(n_experiment_repeats):
    for dataset in datasets:
        X_train, y_train, X_test, y_test, feature_names, \
            compute_features_and_transformer, transform_features, \
            fixed_train_test_split \
            = load_dataset(dataset, experiment_idx,
                           fix_test_shuffle_train=fix_test_shuffle_train)

        if fixed_train_test_split and experiment_idx > 0 and \
                not fix_test_shuffle_train:
            # the dataset has a fixed train/test split so there's no benefit
            # to additional experimental repeats that randomize over the
            # train/test split
            continue

        # load_dataset already shuffles; no need to reshuffle
        proper_train_idx, val_idx = train_test_split(range(len(X_train)),
                                                     test_size=val_ratio,
                                                     shuffle=False)
        if pretrain_frac < 1:
            # sample split (so that `proper_train_idx` corresponds to what the
            # paper calls the "pre-training" data and `true_train_idx`
            # corresponds to what the paper calls the "training" data
            proper_train_idx, true_train_idx = train_test_split(
                    range(len(proper_train_idx)), test_size=pretrain_frac,
                    shuffle=False)
        else:
            true_train_idx = proper_train_idx
        X_proper_train = X_train[proper_train_idx]
        y_proper_train = y_train[proper_train_idx].astype('float32')
        X_val = X_train[val_idx]
        y_val = y_train[val_idx].astype('float32')

        X_proper_train_std, transformer = \
            compute_features_and_transformer(X_proper_train)
        X_val_std = transform_features(X_val, transformer)
        X_proper_train_std = X_proper_train_std.astype('float32')
        X_val_std = X_val_std.astype('float32')

        print('[Dataset: %s (size=%d, raw dim=%d, dim=%d), experiment: %d]'
              % (dataset, len(X_train) + len(X_test), X_train.shape[1],
                 X_proper_train_std.shape[1], experiment_idx))
        print()

        print('-' * 80)
        print('*** Warm-start ***')
        print()

        output_train_metrics_filename \
            = os.path.join(output_dir, 'train',
                           '%s_%s_exp%d_%s_train_metrics_pf%f_%s.txt'
                           % (estimator_name, dataset, experiment_idx,
                              validation_string, pretrain_frac,
                              hyperparam_hash))
        output_best_hyperparam_filename \
            = os.path.join(output_dir, 'train',
                           '%s_%s_exp%d_%s_best_hyperparams_pf%f_%s.pkl'
                           % (estimator_name, dataset, experiment_idx,
                              validation_string, pretrain_frac,
                              hyperparam_hash))
        output_train_timing_filename \
            = output_train_metrics_filename[:-4] + '_time.txt'
        if not os.path.isfile(output_train_metrics_filename) \
                or not os.path.isfile(output_best_hyperparam_filename) \
                or not os.path.isfile(output_train_timing_filename):
            print('Training...', flush=True)
            train_metrics_file = open(output_train_metrics_filename, 'w')
            best_hyperparams = {}

            min_loss = np.inf
            arg_min = None
            arg_min_hyperparam_idx = None
            best_model_filename = 'cache_' + str(uuid.uuid4()) + '.pkl'

            models = []
            train_times = []
            val_times = []

            for hyperparam_idx, hyperparam in enumerate(hyperparams):
                max_features, min_samples_leaf = hyperparam

                # seed different hyperparameters differently to prevent weird
                # behavior where a bad initial seed makes a specific model
                # always look terrible
                hyperparam_random_seed = method_random_seed + hyperparam_idx

                print('Hyperparameter:', (*hyperparam, hyperparam_random_seed))
                print('Fitting RSF model...', flush=True)

                tic = time.time()
                if max_features is None:
                    max_features_parsed = X_proper_train_std.shape[1]
                elif max_features == 'sqrt':
                    max_features_parsed = \
                        max(1, int(np.sqrt(X_proper_train_std.shape[1])))
                elif max_features.endswith('sqrt'):
                    max_features_parsed = \
                        max(1,
                            int(float(max_features[:-4]) *
                                np.sqrt(X_proper_train_std.shape[1])))
                elif max_features == 'log2':
                    max_features_parsed = \
                        max(1,
                            int(np.log2(X_proper_train_std.shape[1])))
                elif max_features.endswith('log2'):
                    max_features_parsed = \
                        max(1,
                            int(float(max_features[:-4]) *
                                np.log2(X_proper_train_std.shape[1])))
                elif type(max_features) is int:
                    max_features_parsed = max_features
                elif type(max_features) is float:
                    max_features_parsed = \
                        max(1, int(max_features * X_proper_train_std.shape[1]))
                else:
                    raise Exception(
                            'Unsupported `max_features` value: '
                            + str(max_features))
                surv_model = \
                    RandomSurvivalForest(
                        n_estimators=100,
                        max_features=max_features_parsed,
                        max_depth=None,
                        oob_score=False,
                        feature_importance=False,
                        min_samples_leaf=min_samples_leaf,
                        random_state=hyperparam_random_seed,
                        n_jobs=n_jobs)
                surv_model.fit(X_proper_train_std, y_proper_train)
                elapsed = time.time() - tic
                train_times.append(elapsed)
                print('- Time elapsed: %f second(s)' % elapsed, flush=True)

                tic = time.time()
                sorted_unique_y_proper_train = np.unique(
                    y_proper_train[:, 0][
                        y_proper_train[:, 1] == 1])
                surv = \
                    surv_model.predict_surv(X_val_std,
                                            sorted_unique_y_proper_train,
                                            presorted_times=True)
                surv_df = pd.DataFrame(surv.T,
                                       columns=range(X_val_std.shape[0]),
                                       index=sorted_unique_y_proper_train)
                val_loss = neg_cindex_td(y_val,
                                         (surv_df.to_numpy(), surv_df.index))
                print('- Val loss:', val_loss)
                if val_loss < min_loss:
                    min_loss = val_loss
                    arg_min = (*hyperparam, hyperparam_random_seed)
                    arg_min_hyperparam_idx = hyperparam_idx
                    surv_model.save(best_model_filename)
                else:
                    del surv_model
                    gc.collect()
                elapsed = time.time() - tic
                val_times.append(elapsed)
                print('- Time elapsed: %f second(s)' % elapsed, flush=True)

            train_metrics_file.close()
            np.savetxt(output_train_timing_filename,
                       np.array([train_times, val_times]).T)

            print('Saving best model found...', flush=True)
            tic = time.time()
            max_features, min_samples_leaf, seed = arg_min
            model_filename = \
                os.path.join(
                    output_dir, 'models',
                    '%s_%s_exp%d_mf%s_msl%d_pf%f_test.pkl'
                    % (estimator_name, dataset, experiment_idx,
                       str(max_features), min_samples_leaf, pretrain_frac))
            os.rename(best_model_filename, model_filename)
            np.savetxt(model_filename[:-4] + '_time.txt',
                       np.array(train_times[
                           arg_min_hyperparam_idx]).reshape(1, -1))

            best_hyperparams['loss'] = (arg_min, min_loss)
            with open(output_best_hyperparam_filename, 'wb') as pickle_file:
                pickle.dump(best_hyperparams, pickle_file,
                            protocol=pickle.HIGHEST_PROTOCOL)
            elapsed = time.time() - tic
            print('- Time elapsed: %f second(s)' % elapsed, flush=True)
        else:
            print('Loading previous validation results...', flush=True)
            with open(output_best_hyperparam_filename, 'rb') as pickle_file:
                best_hyperparams = pickle.load(pickle_file)
            arg_min, min_loss = best_hyperparams['loss']

        print('Best hyperparameters for minimizing loss:',
              arg_min, '-- achieves val loss %f' % min_loss, flush=True)

        max_features, min_samples_leaf, seed = arg_min

        model_filename = \
            os.path.join(
                output_dir, 'models',
                '%s_%s_exp%d_mf%s_msl%d_pf%f_test.pkl'
                % (estimator_name, dataset, experiment_idx,
                   str(max_features), min_samples_leaf, pretrain_frac))
        surv_model = RandomSurvivalForest.load(model_filename)

        print('*** Extracting proximity matrix...')
        X_train_std = transform_features(X_train, transformer)
        tic = time.time()
        prox_filename = model_filename[:-4] + '_prox_matrix.txt'
        time_elapsed_filename = prox_filename[:-4] + '_time.txt'
        if not os.path.isfile(prox_filename):
            leaf_ids = surv_model.predict_leaf_ids(X_train_std)
            n = len(X_train_std)
            prox_matrix = np.eye(n)
            for i in range(n):
                for j in range(i + 1, n):
                    prox_matrix[i, j] = \
                        np.mean(leaf_ids[i] == leaf_ids[j])
                    prox_matrix[j, i] = prox_matrix[i, j]
            elapsed = time.time() - tic
            print('Time elapsed: %f second(s)' % elapsed)
            np.savetxt(time_elapsed_filename,
                       np.array(elapsed).reshape(1, -1))
            np.savetxt(prox_filename, prox_matrix)
        else:
            prox_matrix = np.loadtxt(prox_filename)
            elapsed = float(np.loadtxt(time_elapsed_filename))
            print('Time elapsed (from previous fitting): %f second(s)'
                  % elapsed)

        prox_matrix_time = elapsed

        del surv_model
        gc.collect()

        print('*** Computing MDS embedding...')
        tic = time.time()
        kernel_matrix = np.clip(prox_matrix + 1e-7, 0., 1.)
        rsf_dists = np.sqrt(-np.log(kernel_matrix))
        mds_size = min(len(X_train), X_train.shape[1])
        mds_filename = model_filename[:-4] + '_mds%d.txt' % mds_size
        time_elapsed_filename = mds_filename[:-4] + '_time.txt'
        if not os.path.isfile(mds_filename):
            mds = MDS(n_components=mds_size,
                      metric=True,
                      n_init=mds_n_init,
                      random_state=mds_random_seed,
                      dissimilarity='precomputed')
            mds_embedding = mds.fit_transform(rsf_dists)
            elapsed = time.time() - tic
            print('Time elapsed: %f second(s)' % elapsed)
            np.savetxt(time_elapsed_filename,
                       np.array(elapsed).reshape(1, -1))
            np.savetxt(mds_filename, mds_embedding)
        else:
            mds_embedding = np.loadtxt(mds_filename)
            elapsed = float(np.loadtxt(time_elapsed_filename))
            print('Time elapsed (from previous fitting): %f second(s)'
                  % elapsed)
        mds_embedding = mds_embedding.astype('float32')

        print()

        mds_time = elapsed

        # ---------------------------------------------------------------------

        warmstart_best_params_filename = \
            model_filename[:-6] \
            + '_mds_mlp_params_%s.pkl' \
            % (warmstart_hyperparam_hash)
        warmstart_train_times_filename = \
            model_filename[:-6] \
            + '_mds_training_times_%s.txt' \
            % (warmstart_hyperparam_hash)
        if not os.path.isfile(warmstart_best_params_filename) \
                or not os.path.isfile(warmstart_train_times_filename):
            warmstart_min_score = np.inf
            warmstart_best_params = None
            warmstart_train_times = []
            for warmstart_hyperparam in warmstart_hyperparams:
                squared_radius, n_layers, n_nodes, batch_size, max_n_epochs, \
                    lr = warmstart_hyperparam

                dataset_max_n_epochs = 'max_n_epochs_%s' % dataset
                if dataset_max_n_epochs in config[finetune_method_header]:
                    max_n_epochs = \
                        int(config[finetune_method_header][
                            dataset_max_n_epochs])

                tic = time.time()

                torch.manual_seed(mds_random_seed)
                np.random.seed(mds_random_seed)
                random.seed(mds_random_seed)

                optimizer = tt.optim.Adam(lr=lr)
                if squared_radius == 0:
                    net = tt.practical.MLPVanilla(
                        X_proper_train_std.shape[1],
                        [n_nodes for layer_idx in range(n_layers)],
                        mds_size,
                        True,
                        0.,
                        output_bias=False)
                else:
                    net = tt.practical.MLPVanilla(
                        X_proper_train_std.shape[1],
                        [n_nodes for layer_idx in range(n_layers)],
                        mds_size,
                        True,
                        0.,
                        output_activation=Hypersphere(
                            squared_radius=squared_radius),
                        output_bias=True)
                warmstart_model = tt.Model(net, nn.MSELoss(),
                                           optimizer)
                val_loader = \
                    warmstart_model.make_dataloader(
                        (X_val_std,
                         mds_embedding[val_idx]),
                        batch_size, False)

                warmstart_model_filename = \
                    model_filename[:-6] \
                    + '_mds_mlp_sr%f_nla%d_nno%d_bs%d_mnep%d_lr%f.pt' \
                    % (squared_radius, n_layers, n_nodes, batch_size,
                       max_n_epochs, lr)
                time_elapsed_filename = \
                    warmstart_model_filename[:-3] + '_time.txt'
                if not os.path.isfile(warmstart_model_filename) or \
                        not os.path.isfile(time_elapsed_filename):
                    print('Training warm-start neural net...')
                    train_loader = \
                        warmstart_model.make_dataloader(
                            (X_proper_train_std,
                             mds_embedding[proper_train_idx]),
                            batch_size, True)
                    warmstart_model.fit_dataloader(
                        train_loader, epochs=max_n_epochs,
                        callbacks=[tt.callbacks.EarlyStopping()],
                        val_dataloader=val_loader,
                        verbose=True)
                    elapsed = time.time() - tic
                    print('Time elapsed: %f second(s)' % elapsed, flush=True)
                    warmstart_model.save_model_weights(
                        warmstart_model_filename)
                    np.savetxt(time_elapsed_filename,
                               np.array(elapsed).reshape(1, -1))
                    warmstart_train_times.append(elapsed)
                    score = warmstart_model.score_in_batches_dataloader(
                        val_loader)['loss']
                    print(warmstart_hyperparam, ':', score, flush=True)
                    if score < warmstart_min_score:
                        warmstart_min_score = score
                        warmstart_best_params = \
                            (squared_radius, n_layers, n_nodes, batch_size,
                             max_n_epochs, lr)
                else:
                    print('Loading previously fitted results...')
                    warmstart_model.load_model_weights(
                        warmstart_model_filename)
                    elapsed = float(np.loadtxt(time_elapsed_filename))
                    print('Time elapsed (from previous fitting): %f second(s)'
                          % elapsed, flush=True)
                    warmstart_train_times.append(elapsed)
                    score = warmstart_model.score_in_batches_dataloader(
                        val_loader)['loss']
                    print(warmstart_hyperparam, ':', score, flush=True)
                    if score < warmstart_min_score:
                        warmstart_min_score = score
                        warmstart_best_params = \
                            (squared_radius, n_layers, n_nodes, batch_size,
                             max_n_epochs, lr)
                print()

            with open(warmstart_best_params_filename, 'wb') as pickle_file:
                pickle.dump(warmstart_best_params, pickle_file,
                            protocol=pickle.HIGHEST_PROTOCOL)
            np.savetxt(warmstart_train_times_filename,
                       np.array(warmstart_train_times))

        print('Loading warm-start neural net...')
        with open(warmstart_best_params_filename, 'rb') as pickle_file:
            warmstart_best_params = pickle.load(pickle_file)
            squared_radius, n_layers, n_nodes, batch_size, max_n_epochs, lr \
                = warmstart_best_params
        torch.manual_seed(mds_random_seed)
        np.random.seed(mds_random_seed)
        random.seed(mds_random_seed)

        optimizer = tt.optim.Adam(lr=lr)
        if squared_radius == 0:
            net = tt.practical.MLPVanilla(
                X_proper_train_std.shape[1],
                [n_nodes for layer_idx in range(n_layers)],
                mds_size,
                True,
                0.,
                output_bias=False)
        else:
            net = tt.practical.MLPVanilla(
                X_proper_train_std.shape[1],
                [n_nodes for layer_idx in range(n_layers)],
                mds_size,
                True,
                0.,
                output_activation=Hypersphere(squared_radius=squared_radius),
                output_bias=True)
        warmstart_model = tt.Model(net, nn.MSELoss(), optimizer)

        warmstart_model_filename = \
            model_filename[:-6] \
            + '_mds_mlp_sr%f_nla%d_nno%d_bs%d_mnep%d_lr%f.pt' \
            % (squared_radius, n_layers, n_nodes, batch_size, max_n_epochs, lr)
        warmstart_model.load_model_weights(warmstart_model_filename)

        warmstart_train_times = \
            np.loadtxt(warmstart_train_times_filename)
        print('Train times mean (std): %f (%f)'
              % (np.mean(warmstart_train_times),
                 np.std(warmstart_train_times)))

        del warmstart_model
        gc.collect()

        print('Warm-start hyperparameters:', warmstart_best_params)

        print()

        # ---------------------------------------------------------------------

        print('-' * 80)
        print('*** Fine-tuning kernel function ***')
        print()

        output_train_metrics_filename \
            = os.path.join(output_dir, 'train',
                           '%s_%s_exp%d_%s_train_metrics_%s.txt'
                           % (full_estimator_name, dataset, experiment_idx,
                              validation_string, finetune_hyperparam_hash))
        output_best_hyperparam_filename \
            = os.path.join(output_dir, 'train',
                           '%s_%s_exp%d_%s_best_hyperparams_%s.pkl'
                           % (full_estimator_name, dataset, experiment_idx,
                              validation_string, finetune_hyperparam_hash))
        if not os.path.isfile(output_train_metrics_filename) or \
                not os.path.isfile(output_best_hyperparam_filename):
            print('Training...', flush=True)
            train_metrics_file = open(output_train_metrics_filename, 'w')
            best_hyperparams = {}

            min_loss = np.inf
            arg_min = None
            best_model_filename = None

            for hyperparam_idx, hyperparam in enumerate(finetune_hyperparams):
                pretrain_frac, n_durations, batch_size, max_n_epochs, \
                    lr = hyperparam

                # seed different hyperparameters differently to prevent weird
                # behavior where a bad initial seed makes a specific model
                # always look terrible
                hyperparam_random_seed = finetune_method_random_seed \
                    + hyperparam_idx

                dataset_max_n_epochs = 'max_n_epochs_%s' % dataset
                if dataset_max_n_epochs in config[finetune_method_header]:
                    max_n_epochs = \
                        int(config[finetune_method_header][
                            dataset_max_n_epochs])

                tic = time.time()
                torch.manual_seed(hyperparam_random_seed)
                np.random.seed(hyperparam_random_seed)
                random.seed(hyperparam_random_seed)

                optimizer = tt.optim.Adam(lr=lr)
                if squared_radius == 0:
                    net = tt.practical.MLPVanilla(
                        X_proper_train_std.shape[1],
                        [n_nodes for layer_idx in range(n_layers)],
                        mds_size,
                        True,
                        0.,
                        output_bias=False)
                else:
                    net = tt.practical.MLPVanilla(
                        X_proper_train_std.shape[1],
                        [n_nodes for layer_idx in range(n_layers)],
                        mds_size,
                        True,
                        0.,
                        output_activation=Hypersphere(
                            squared_radius=squared_radius),
                        output_bias=True)

                if n_durations == 0:
                    label_transformer = LabTransDiscreteTime(
                        np.unique(y_proper_train[:, 0][
                            y_proper_train[:, 1] == 1]))
                else:
                    label_transformer = LabTransDiscreteTime(
                        n_durations, scheme='quantiles')
                y_proper_train_discrete = \
                    label_transformer.fit_transform(*y_proper_train.T)

                model = NKSDiscrete(net,
                                    optimizer,
                                    duration_index=label_transformer.cuts)
                model.load_model_weights(warmstart_model_filename)
                model.net.train()

                finetune_model_filename = \
                    os.path.join(
                        output_dir, 'models',
                        '%s_%s_exp%d_pf%f_nd%d_sr%f_'
                        % (full_estimator_name, dataset, experiment_idx,
                           pretrain_frac, n_durations, squared_radius)
                        +
                        'nla%d_nno%d_bs%d_mnep%d_lr%f.pt'
                        % (n_layers, n_nodes, batch_size, max_n_epochs, lr))
                time_elapsed_filename = \
                    finetune_model_filename[:-3] + '_time.txt'
                epoch_time_elapsed_filename = \
                    finetune_model_filename[:-3] + '_epoch_times.txt'
                epoch_times = []

                train_pair = (X_proper_train_std, y_proper_train_discrete)
                train_loader = \
                    model.make_dataloader(train_pair, batch_size, True)
                model.training_data = \
                    (torch.tensor(X_proper_train_std),
                     (torch.tensor(y_proper_train_discrete[0],
                                   dtype=torch.int64,
                                   device=model.device),
                      torch.tensor(y_proper_train_discrete[1],
                                   dtype=torch.float32,
                                   device=model.device)))
                model.duration_index = label_transformer.cuts
                best_loss = np.inf
                for epoch_idx in range(max_n_epochs):
                    tic_ = time.time()
                    model.fit_dataloader(train_loader, epochs=1)
                    epoch_train_time = time.time() - tic_
                    tic_ = time.time()
                    # try:
                    model.train_embeddings = model.predict(X_proper_train_std,
                                                           batch_size, False,
                                                           True, False, False, 0)
                    model.train_embeddings.to(model.device)
                    surv_df = \
                        model.interpolate(10).predict_surv_df(
                            X_val_std)
                    y_val_pred = (surv_df.to_numpy(), surv_df.index)
                    val_loss = neg_cindex_td(y_val, y_val_pred, exact=False)
                    # except:
                    #     val_loss = 0.
                    epoch_val_time = time.time() - tic_
                    epoch_times.append([epoch_train_time,
                                        epoch_val_time])
                    if val_loss != 0.:
                        new_hyperparam = \
                            (pretrain_frac, n_durations,
                             batch_size, epoch_idx + 1, lr,
                             hyperparam_random_seed)
                        print(new_hyperparam,
                              '--',
                              'val loss %f' % val_loss,
                              '--',
                              'train time %f sec(s)'
                              % epoch_train_time,
                              '--',
                              'val time %f sec(s)' % epoch_val_time,
                              flush=True)
                        print(new_hyperparam, ':', val_loss, flush=True,
                              file=train_metrics_file)

                        if val_loss < best_loss:
                            best_loss = val_loss
                            wait_idx = 0
                            model.save_net(finetune_model_filename)

                            if val_loss < min_loss:
                                min_loss = val_loss
                                arg_min = new_hyperparam
                                best_model_filename = \
                                    finetune_model_filename
                        else:
                            wait_idx += 1
                            if patience > 0 and wait_idx >= patience:
                                break
                    else:
                        # weird corner case
                        break

                np.savetxt(epoch_time_elapsed_filename,
                           np.array(epoch_times))

                elapsed = time.time() - tic
                print('Time elapsed: %f second(s)' % elapsed, flush=True)
                np.savetxt(time_elapsed_filename,
                           np.array(elapsed).reshape(1, -1))

                del model
                gc.collect()

            train_metrics_file.close()

            best_hyperparams['loss'] = (arg_min, min_loss)
            with open(output_best_hyperparam_filename, 'wb') as pickle_file:
                pickle.dump(best_hyperparams, pickle_file,
                            protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('Loading previous validation results...', flush=True)
            with open(output_best_hyperparam_filename, 'rb') as pickle_file:
                best_hyperparams = pickle.load(pickle_file)
            arg_min, min_loss = best_hyperparams['loss']

        print('Best hyperparameters for minimizing loss:',
              arg_min, '-- achieves val loss %f' % min_loss, flush=True)

        print()

        X_true_train = X_train[true_train_idx]
        X_true_train_std = transform_features(X_true_train, transformer)
        X_true_train_std = X_true_train_std.astype('float32')
        y_true_train = y_train[true_train_idx].astype('float32')

        pretrain_frac, n_durations, batch_size, n_epochs, lr, seed \
            = arg_min

        if n_durations == 0:
            label_transformer = LabTransDiscreteTime(
                np.unique(y_proper_train[:, 0][
                    y_proper_train[:, 1] == 1]))
        else:
            label_transformer = LabTransDiscreteTime(
                n_durations, scheme='quantiles')
        y_proper_train_discrete = \
            label_transformer.fit_transform(*y_proper_train.T)

        tic = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        optimizer = tt.optim.Adam(lr=lr)
        if squared_radius == 0:
            net = tt.practical.MLPVanilla(
                X_proper_train_std.shape[1],
                [n_nodes for layer_idx in range(n_layers)],
                mds_size,
                True,
                0.,
                output_bias=False)
        else:
            net = tt.practical.MLPVanilla(
                X_proper_train_std.shape[1],
                [n_nodes for layer_idx in range(n_layers)],
                mds_size,
                True,
                0.,
                output_activation=Hypersphere(squared_radius=squared_radius),
                output_bias=True)

        model = NKSDiscrete(net,
                            optimizer,
                            duration_index=label_transformer.cuts)
        model.load_model_weights(warmstart_model_filename)
        model.net.train()

        max_n_epochs = int(config[finetune_method_header]['max_n_epochs'])
        dataset_max_n_epochs = 'max_n_epochs_%s' % dataset
        if dataset_max_n_epochs in config[finetune_method_header]:
            max_n_epochs = \
                int(config[finetune_method_header][
                    dataset_max_n_epochs])

        model_filename = \
            os.path.join(
                output_dir, 'models',
                '%s_%s_exp%d_pf%f_nd%d_sr%f_'
                % (full_estimator_name, dataset, experiment_idx, pretrain_frac,
                   n_durations, squared_radius)
                +
                'nla%d_nno%d_bs%d_mnep%d_lr%f.pt'
                % (n_layers, n_nodes, batch_size, max_n_epochs, lr))
        model.load_net(model_filename)

        time_elapsed_filename = model_filename[:-3] + '_time.txt'
        elapsed = float(np.loadtxt(time_elapsed_filename))
        print('Time elapsed (from previous fitting): %f second(s)'
              % elapsed, flush=True)

        print()

        # ---------------------------------------------------------------------
        # Test set prediction
        #

        print('Testing...', flush=True)
        X_test_std = transform_features(X_test, transformer)
        X_test_std = X_test_std.astype('float32')
        y_test = y_test.astype('float32')

        model.training_data = \
            (torch.tensor(X_proper_train_std),
             (torch.tensor(y_proper_train_discrete[0],
                           dtype=torch.int64,
                           device=model.device),
              torch.tensor(y_proper_train_discrete[1],
                           dtype=torch.float32,
                           device=model.device)))
        model.train_embeddings = model.predict(X_proper_train_std,
                                               batch_size, False,
                                               True, False, False, 0)
        model.train_embeddings.to(model.device)
        surv_df = model.interpolate(10).predict_surv_df(X_test_std)
        y_test_pred = (surv_df.to_numpy(), surv_df.index)

        loss = neg_cindex_td(y_test, y_test_pred)
        print('Hyperparameter', arg_min, 'achieves test loss %f' % loss,
              flush=True)

        final_test_scores = {}
        test_set_metrics = [loss]

        if not compute_bootstrap_CI:
            final_test_scores[arg_min] = tuple(test_set_metrics)
        else:
            rng = np.random.RandomState(bootstrap_random_seed)

            bootstrap_dir = \
                os.path.join(
                    output_dir, 'bootstrap',
                    '%s_%s_exp%d_pf%f_nd%d_sr%f_'
                    % (full_estimator_name, dataset, experiment_idx,
                       pretrain_frac, n_durations, squared_radius)
                    +
                    'nla%d_nno%d_bs%d_nep%d_lr%f_test'
                    % (n_layers, n_nodes, batch_size, n_epochs, lr))
            os.makedirs(bootstrap_dir, exist_ok=True)
            bootstrap_losses_filename = os.path.join(bootstrap_dir,
                                                     'bootstrap_losses.txt')
            if not os.path.isfile(bootstrap_losses_filename):
                bootstrap_losses = []
                for bootstrap_idx in range(bootstrap_n_samples):
                    bootstrap_sample_indices = \
                        rng.choice(X_test_std.shape[0],
                                   size=X_test_std.shape[0],
                                   replace=True)

                    surv_pred, times_pred = y_test_pred
                    y_bootstrap_pred = \
                        (surv_pred.T[bootstrap_sample_indices].T,
                         times_pred)

                    bootstrap_losses.append(
                        neg_cindex_td(y_test[bootstrap_sample_indices],
                                      y_bootstrap_pred,
                                      exact=False))

                bootstrap_losses = np.array(bootstrap_losses)
                np.savetxt(bootstrap_losses_filename, bootstrap_losses)
            else:
                bootstrap_losses = \
                    np.loadtxt(bootstrap_losses_filename).flatten()
            print()

            sorted_bootstrap_losses = np.sort(bootstrap_losses)

            tail_prob = ((1. - bootstrap_CI_coverage) / 2)
            lower_end = int(np.floor(tail_prob * bootstrap_n_samples))
            upper_end = int(np.ceil((1. - tail_prob) * bootstrap_n_samples))
            print('%0.1f%% bootstrap confidence intervals:'
                  % (100 * bootstrap_CI_coverage), flush=True)
            print('loss: (%0.8f, %0.8f)'
                  % (sorted_bootstrap_losses[lower_end],
                     sorted_bootstrap_losses[upper_end]), flush=True)
            print()
            test_set_metrics += [sorted_bootstrap_losses[lower_end],
                                 sorted_bootstrap_losses[upper_end]]

            np.savetxt(os.path.join(bootstrap_dir, 'final_metrics.txt'),
                       np.array(test_set_metrics))

            final_test_scores[arg_min] = tuple(test_set_metrics)

            del model
            gc.collect()

        if compute_bootstrap_CI:
            test_csv_writer.writerow(
                [dataset, experiment_idx, full_estimator_name,
                 final_test_scores[arg_min][0],
                 final_test_scores[arg_min][1],
                 final_test_scores[arg_min][2]])
        else:
            test_csv_writer.writerow(
                [dataset, experiment_idx, full_estimator_name,
                 final_test_scores[arg_min][0]])

        print()
        print()
