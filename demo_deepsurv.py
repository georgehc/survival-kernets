#!/usr/bin/env python
"""
DeepSurv baseline (uses PyCox's DeepSurv implementation)

Author: George H. Chen (georgechen [at symbol] cmu.edu)
"""
import ast
import configparser
import csv
import gc
import hashlib
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import pickle
import random
import shutil
import sys
import time
import uuid

import numpy as np
import torch
# torch.set_deterministic(True)  # causes problems with survival analysis
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from sklearn.model_selection import train_test_split

import torchtuples as tt

from datasets import load_dataset
from metrics import neg_cindex
from models import Model, CoxPH


estimator_name = 'deepsurv'

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
verbose = int(config['DEFAULT']['verbose']) > 0

compute_bootstrap_CI = int(config['DEFAULT']['compute_bootstrap_CI']) > 0
bootstrap_CI_coverage = float(config['DEFAULT']['bootstrap_CI_coverage'])
bootstrap_n_samples = int(config['DEFAULT']['bootstrap_n_samples'])
bootstrap_random_seed = int(config['DEFAULT']['bootstrap_random_seed'])

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)

max_n_epochs = int(config[method_header]['max_n_epochs'])

batch_norm = True

hyperparams = \
    [(batch_size, max_n_epochs, n_layers, n_nodes, lr)
     for batch_size
     in ast.literal_eval(config[method_header]['batch_size'])
     for n_layers
     in ast.literal_eval(config[method_header]['n_layers'])
     for n_nodes
     in ast.literal_eval(config[method_header]['n_nodes'])
     for lr
     in ast.literal_eval(config[method_header]['learning_rate'])]

hyperparam_hash = hashlib.sha256()
hyperparam_hash.update(str(hyperparams).encode('utf-8'))
hyperparam_hash = hyperparam_hash.hexdigest()

validation_string = 'vr%f' % val_ratio

output_test_table_filename \
    = os.path.join(output_dir,
                   '%s_experiments%d_%s_test_metrics_%s.csv'
                   % (estimator_name,
                      n_experiment_repeats,
                      validation_string,
                      hyperparam_hash))
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
                fixed_train_test_split = \
            load_dataset(dataset, experiment_idx,
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
        X_proper_train = X_train[proper_train_idx]
        y_proper_train = y_train[proper_train_idx].astype('float32')
        X_val = X_train[val_idx]
        y_val = y_train[val_idx].astype('float32')

        X_proper_train_std, transformer = \
            compute_features_and_transformer(X_proper_train)
        X_val_std = transform_features(X_val, transformer)
        X_proper_train_std = X_proper_train_std.astype('float32')
        X_val_std = X_val_std.astype('float32')

        print('[Dataset: %s, experiment: %d]' % (dataset, experiment_idx))
        print()

        output_train_metrics_filename \
            = os.path.join(output_dir, 'train',
                           '%s_%s_exp%d_%s_train_metrics_%s.txt'
                           % (estimator_name, dataset, experiment_idx,
                              validation_string, hyperparam_hash))
        output_best_hyperparam_filename \
            = os.path.join(output_dir, 'train',
                           '%s_%s_exp%d_%s_best_hyperparams_%s.pkl'
                           % (estimator_name, dataset, experiment_idx,
                              validation_string, hyperparam_hash))
        if not os.path.isfile(output_train_metrics_filename) or \
                not os.path.isfile(output_best_hyperparam_filename):
            print('Training...', flush=True)
            train_metrics_file = open(output_train_metrics_filename, 'w')
            best_hyperparams = {}

            min_loss = np.inf
            arg_min = None
            best_model_filename = 'cache_' + str(uuid.uuid4()) + '.pt'

            for hyperparam_idx, hyperparam in enumerate(hyperparams):
                batch_size, max_n_epochs, n_layers, n_nodes, lr = hyperparam

                # seed different hyperparameters differently to prevent weird
                # behavior where a bad initial seed makes a specific model
                # always look terrible
                hyperparam_random_seed = method_random_seed + hyperparam_idx

                dataset_max_n_epochs = 'max_n_epochs_%s' % dataset
                if dataset_max_n_epochs in config[method_header]:
                    max_n_epochs = \
                        int(config[method_header][dataset_max_n_epochs])

                tic = time.time()
                torch.manual_seed(hyperparam_random_seed)
                np.random.seed(hyperparam_random_seed)
                random.seed(hyperparam_random_seed)

                optimizer = tt.optim.Adam(lr=lr)
                net = tt.practical.MLPVanilla(X_proper_train_std.shape[1],
                                              [n_nodes for layer_idx
                                               in range(n_layers)],
                                              1,
                                              batch_norm,
                                              0.,
                                              output_bias=False)
                model = CoxPH(net, optimizer)

                time_elapsed_filename = \
                    os.path.join(output_dir, 'models',
                                 '%s_%s_exp%d_bs%d_mnep%d_nla%d_nno%d_'
                                 % (estimator_name, dataset, experiment_idx,
                                    batch_size, max_n_epochs, n_layers,
                                    n_nodes)
                                 +
                                 'lr%f_time.txt'
                                 % lr)
                epoch_time_elapsed_filename = \
                    time_elapsed_filename[:-8] + 'epoch_times.txt'
                epoch_times = []

                train_pair = (X_proper_train_std,
                              (y_proper_train[:, 0],
                               y_proper_train[:, 1]))
                model.training_data = tt.tuplefy(*train_pair)
                train_loader = \
                    model.make_dataloader(train_pair, batch_size, True)
                best_loss = np.inf
                for epoch_idx in range(max_n_epochs):
                    tic_ = time.time()
                    model.fit_dataloader(train_loader, epochs=1)
                    epoch_train_time = time.time() - tic_
                    tic_ = time.time()
                    y_val_pred = model.predict(X_val_std).flatten()
                    val_loss = neg_cindex(y_val, y_val_pred)
                    epoch_val_time = time.time() - tic_
                    epoch_times.append([epoch_train_time,
                                        epoch_val_time])
                    if val_loss != 0.:
                        new_hyperparam = \
                            (batch_size, epoch_idx + 1, n_layers,
                             n_nodes, lr, hyperparam_random_seed)
                        print(new_hyperparam,
                              '--',
                              'val loss %f' % val_loss,
                              '--',
                              'train time %f sec(s)' % epoch_train_time,
                              '--',
                              'val time %f sec(s)' % epoch_val_time,
                              flush=True)
                        print(new_hyperparam, ':', val_loss, flush=True,
                              file=train_metrics_file)

                        if val_loss < best_loss:
                            best_loss = val_loss
                            wait_idx = 0

                            if val_loss < min_loss:
                                min_loss = val_loss
                                arg_min = new_hyperparam
                                model.save_net(best_model_filename)
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

            batch_size, n_epochs, n_layers, n_nodes, lr, seed = arg_min
            model_filename = \
                os.path.join(output_dir, 'models',
                             '%s_%s_exp%d_bs%d_nep%d_nla%d_nno%d_lr%f_test.pt'
                             % (estimator_name, dataset, experiment_idx,
                                batch_size, n_epochs, n_layers, n_nodes, lr))
            time_elapsed_filename = \
                os.path.join(output_dir, 'models',
                             '%s_%s_exp%d_bs%d_mnep%d_nla%d_nno%d_'
                             % (estimator_name, dataset, experiment_idx,
                                batch_size, max_n_epochs, n_layers,
                                n_nodes)
                             +
                             'lr%f_time.txt'
                             % lr)
            os.rename(best_model_filename, model_filename)
            shutil.copy(time_elapsed_filename,
                        model_filename[:-3] + '_time.txt')

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
        print('Testing...', flush=True)
        X_test_std = transform_features(X_test, transformer)
        X_test_std = X_test_std.astype('float32')
        y_test = y_test.astype('float32')
        final_test_scores = {}

        batch_size, n_epochs, n_layers, n_nodes, lr, seed = arg_min

        tic = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        optimizer = tt.optim.Adam(lr=lr)
        net = tt.practical.MLPVanilla(X_proper_train_std.shape[1],
                                      [n_nodes for layer_idx
                                       in range(n_layers)],
                                      1,
                                      batch_norm,
                                      0.,
                                      output_bias=False)
        model = CoxPH(net, optimizer)

        model_filename = \
            os.path.join(output_dir, 'models',
                         '%s_%s_exp%d_bs%d_nep%d_nla%d_nno%d_lr%f_test.pt'
                         % (estimator_name, dataset, experiment_idx,
                            batch_size, n_epochs, n_layers, n_nodes, lr))
        time_elapsed_filename = model_filename[:-3] + '_time.txt'
        model.load_net(model_filename)
        elapsed = float(np.loadtxt(time_elapsed_filename))
        print('Time elapsed (from previous fitting): %f second(s)'
              % elapsed, flush=True)

        y_test_pred = model.predict(X_test_std).flatten()

        loss = neg_cindex(y_test, y_test_pred)
        print('Hyperparameter', arg_min, 'achieves test loss %f' % loss,
              flush=True)

        test_set_metrics = [loss]

        if not compute_bootstrap_CI:
            final_test_scores[arg_min] = tuple(test_set_metrics)
        else:
            rng = np.random.RandomState(bootstrap_random_seed)

            bootstrap_dir = \
                os.path.join(output_dir, 'bootstrap',
                             '%s_%s_exp%d_bs%d_nep%d_nla%d_nno%d_lr%f_test'
                             % (estimator_name, dataset, experiment_idx,
                                batch_size, n_epochs, n_layers, n_nodes, lr))
            os.makedirs(bootstrap_dir, exist_ok=True)
            bootstrap_losses_filename = os.path.join(bootstrap_dir,
                                                     'bootstrap_losses.txt')
            if not os.path.isfile(bootstrap_losses_filename):
                bootstrap_losses = []
                for bootstrap_idx in range(bootstrap_n_samples):
                    bootstrap_sample_indices = \
                        rng.choice(X_test.shape[0],
                                   size=X_test.shape[0],
                                   replace=True)

                    y_bootstrap_pred = y_test_pred[bootstrap_sample_indices]

                    bootstrap_losses.append(
                        neg_cindex(y_test[bootstrap_sample_indices],
                                   y_bootstrap_pred))

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
                [dataset, experiment_idx, estimator_name,
                 final_test_scores[arg_min][0],
                 final_test_scores[arg_min][1],
                 final_test_scores[arg_min][2]])
        else:
            test_csv_writer.writerow(
                [dataset, experiment_idx, estimator_name,
                 final_test_scores[arg_min][0]])
        output_test_table_file.flush()

        print()
        print()

output_test_table_file.close()
