#!/usr/bin/env python
"""
XGBoost baseline (uses the official XGBoost implementation with Cox loss)

Author: George H. Chen (georgechen [at symbol] cmu.edu)
"""
import ast
import configparser
import csv
import gc
import hashlib
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import pickle
import shutil
import sys
import time

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from metrics import neg_cindex


estimator_name = 'xgb'

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
max_n_cores = int(config['DEFAULT']['max_n_cores'])

compute_bootstrap_CI = int(config['DEFAULT']['compute_bootstrap_CI']) > 0
bootstrap_CI_coverage = float(config['DEFAULT']['bootstrap_CI_coverage'])
bootstrap_n_samples = int(config['DEFAULT']['bootstrap_n_samples'])
bootstrap_random_seed = int(config['DEFAULT']['bootstrap_random_seed'])

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)

if max_n_cores <= 0:
    n_jobs = os.cpu_count()
else:
    n_jobs = min(max_n_cores, os.cpu_count())

max_n_rounds = int(config[method_header]['max_n_rounds'])

pretrain_frac = float(config[method_header]['pretrain_frac'])
hyperparams = \
    [(n_parallel_trees, max_features, max_depth, subsample, eta, max_n_rounds)
     for n_parallel_trees
     in ast.literal_eval(config[method_header]['n_parallel_trees'])
     for max_features
     in ast.literal_eval(config[method_header]['max_features'])
     for max_depth
     in ast.literal_eval(config[method_header]['max_depth'])
     for subsample
     in ast.literal_eval(config[method_header]['subsample'])
     for eta
     in ast.literal_eval(config[method_header]['eta'])]

hyperparam_hash = hashlib.sha256()
hyperparam_hash.update(str(hyperparams).encode('utf-8'))
hyperparam_hash = hyperparam_hash.hexdigest()

validation_string = 'vr%f' % val_ratio

output_test_table_filename \
    = os.path.join(output_dir,
                   '%s_experiments%d_%s_test_metrics_pf%f_%s.csv'
                   % (estimator_name,
                      n_experiment_repeats,
                      validation_string,
                      pretrain_frac,
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
        if pretrain_frac < 1:
            # further split the proper training set; this functionality is
            # provided in support of deep kernets' TUNA warm-start procedure
            # (to make sure base neural net training never looks at a subset
            # of the data considered to be the "true" training data)
            proper_train_idx, true_train_idx = train_test_split(
                    range(len(proper_train_idx)), test_size=pretrain_frac,
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

        xgb_params = {'nthread': n_jobs,
                      'tree_method': 'gpu_hist',
                      'objective': 'survival:cox'}

        # for the Cox model, XGBoost expects censored
        # observations to be specified as negative
        dtrain = xgb.DMatrix(X_proper_train_std,
                             label=y_proper_train[:, 0] *
                                   2*(y_proper_train[:, 1] - 0.5))
        dval = xgb.DMatrix(X_val_std)

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
            best_model_idx = None
            prev_best_model_idx = None

            models = []
            train_times = []
            val_times = []

            for hyperparam_idx, hyperparam in enumerate(hyperparams):
                n_parallel_trees, max_features, max_depth, subsample, eta, \
                    max_n_rounds = hyperparam

                dataset_max_n_rounds = 'max_n_rounds_%s' % dataset
                if dataset_max_n_rounds in config[method_header]:
                    max_n_rounds = \
                        int(config[method_header][
                            dataset_max_n_rounds])

                # seed different hyperparameters differently to prevent weird
                # behavior where a bad initial seed makes a specific model
                # always look terrible
                hyperparam_random_seed = method_random_seed + hyperparam_idx

                print('Hyperparameter:', (*hyperparam, hyperparam_random_seed))
                print('Fitting XGBoost model...', flush=True)

                tic = time.time()

                xgb_params['seed'] = hyperparam_random_seed

                if max_features is None:
                    xgb_params['colsample_bynode'] = 1.
                elif max_features == 'sqrt':
                    xgb_params['colsample_bynode'] = \
                        np.sqrt(X_proper_train_std.shape[1]) / \
                        X_proper_train_std.shape[1]
                elif max_features.endswith('sqrt'):
                    xgb_params['colsample_bynode'] = \
                        min(float(max_features[:-4]) *
                            np.sqrt(X_proper_train_std.shape[1]) /
                            X_proper_train_std.shape[1], 1)
                elif max_features == 'log2':
                    xgb_params['colsample_bynode'] = \
                        np.log2(X_proper_train_std.shape[1]) / \
                        X_proper_train_std.shape[1]
                elif max_features.endswith('log2'):
                    xgb_params['colsample_bynode'] = \
                        min(float(max_features[:-4]) *
                            np.log2(X_proper_train_std.shape[1]) /
                            X_proper_train_std.shape[1], 1)
                elif type(max_features) is int:
                    xgb_params['colsample_bynode'] = \
                        max_features / X_proper_train_std.shape[1]
                elif type(max_features) is float:
                    xgb_params['colsample_bynode'] = max_features
                else:
                    raise Exception(
                            'Unsupported `max_features` value: '
                            + str(max_features))

                xgb_params['num_parallel_tree'] = n_parallel_trees
                xgb_params['max_depth'] = max_depth
                xgb_params['subsample'] = subsample
                xgb_params['eta'] = eta
                models.append(xgb.train(xgb_params, dtrain, max_n_rounds))
                elapsed = time.time() - tic
                train_times.append(elapsed)
                print('- Time elapsed: %f second(s)' % elapsed, flush=True)

                print('Using validation set to choose number of rounds...')
                tic = time.time()
                prev_best_model_idx = best_model_idx
                for n_rounds in range(1, max_n_rounds + 1):
                    y_val_pred = models[-1][:n_rounds].predict(dval)
                    try:
                        val_loss = neg_cindex(y_val, y_val_pred)
                    except:
                        val_loss = np.inf

                    new_hyperparam = (*hyperparam[:-1], n_rounds,
                                      hyperparam_random_seed)
                    print(new_hyperparam, ':', val_loss, flush=True)
                    print(new_hyperparam, ':', val_loss, flush=True,
                          file=train_metrics_file)

                    if val_loss < min_loss:
                        min_loss = val_loss
                        arg_min = new_hyperparam
                        best_model_idx = hyperparam_idx
                elapsed = time.time() - tic
                val_times.append(elapsed)
                print('- Time elapsed: %f second(s)' % elapsed, flush=True)

                if best_model_idx == hyperparam_idx:
                    # there's been an update to which model is best so far
                    if prev_best_model_idx is not None:
                        _ = models[prev_best_model_idx]
                        models[prev_best_model_idx] = None
                        _.__del__()
                        del _
                        gc.collect()
                else:
                    # most recent model need not be retained
                    _ = models[-1]
                    models[-1] = None
                    _.__del__()
                    del _
                    gc.collect()

            train_metrics_file.close()
            np.savetxt(output_train_timing_filename,
                       np.array([train_times, val_times]).T)

            print('Saving best model found...', flush=True)
            tic = time.time()
            n_parallel_trees, max_features, max_depth, subsample, eta, \
                n_rounds, seed = arg_min
            model_filename = \
                os.path.join(
                    output_dir, 'models',
                    '%s_%s_exp%d_t%d_mf%s_md%d_s%f_e%f_nr%d_pf%f_test.model'
                    % (estimator_name, dataset, experiment_idx,
                       n_parallel_trees, str(max_features), max_depth,
                       subsample, eta, n_rounds, pretrain_frac))
            best_model = models[best_model_idx]
            best_model.save_model(model_filename)

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

        print()
        print('Testing...', flush=True)
        X_test_std = transform_features(X_test, transformer)
        dtest = xgb.DMatrix(X_test_std)
        final_test_scores = {}

        n_parallel_trees, max_features, max_depth, subsample, eta, \
            n_rounds, seed = arg_min

        tic = time.time()
        model_filename = \
            os.path.join(
                output_dir, 'models',
                '%s_%s_exp%d_t%d_mf%s_md%d_s%f_e%f_nr%d_pf%f_test.model'
                % (estimator_name, dataset, experiment_idx, n_parallel_trees,
                   str(max_features), max_depth, subsample, eta, n_rounds,
                   pretrain_frac))

        model = xgb.Booster({'nthread': n_jobs})
        model.load_model(model_filename)

        y_test_pred = model[:(n_rounds * n_parallel_trees)].predict(dtest)
        loss = neg_cindex(y_test, y_test_pred)
        print('Hyperparameter', arg_min, 'achieves test loss %f' % loss,
              flush=True)

        test_set_metrics = [loss]

        if not compute_bootstrap_CI:
            final_test_scores[arg_min] = tuple(test_set_metrics)
        else:
            rng = np.random.RandomState(bootstrap_random_seed)

            bootstrap_dir = \
                os.path.join(
                    output_dir, 'bootstrap',
                    '%s_%s_exp%d_t%d_mf%s_md%d_s%f_e%f_nr%d_pf%f_test'
                    % (estimator_name, dataset, experiment_idx,
                       n_parallel_trees, str(max_features), max_depth,
                       subsample, eta, n_rounds, pretrain_frac))
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

        print()
        print()
