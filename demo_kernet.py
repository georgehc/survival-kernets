#!/usr/bin/env python
"""
Survival kernets without TUNA warm-start

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
from metrics import neg_cindex_td
from models import NKS, Hypersphere, Model, NKSSummary, NKSSummaryLoss
from pycox.preprocessing.label_transforms import LabTransDiscreteTime


estimator_name = 'kernet'

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

n_neighbors_range = ast.literal_eval(config['DEFAULT']['ANN_max_n_neighbors'])

compute_bootstrap_CI = int(config['DEFAULT']['compute_bootstrap_CI']) > 0
bootstrap_CI_coverage = float(config['DEFAULT']['bootstrap_CI_coverage'])
bootstrap_n_samples = int(config['DEFAULT']['bootstrap_n_samples'])
bootstrap_random_seed = int(config['DEFAULT']['bootstrap_random_seed'])

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)

max_n_epochs = int(config[method_header]['max_n_epochs'])

pretrain_frac = float(config[method_header]['pretrain_frac'])

finetune_summaries = int(config[method_header]['finetune_summaries'])
sigma_range = ast.literal_eval(config[method_header]['sigma'])
n_durations_range = ast.literal_eval(config[method_header]['n_durations'])
gamma_range = ast.literal_eval(config[method_header]['gamma'])
lr_range = ast.literal_eval(config[method_header]['learning_rate'])
hyperparams = \
    [(pretrain_frac, alpha, sigma, n_durations, gamma, beta, min_kernel_weight,
      squared_radius, n_neighbors, n_layers, n_nodes, batch_size, max_n_epochs,
      lr)
     for alpha
     in ast.literal_eval(config[method_header]['alpha'])
     for sigma
     in sigma_range
     for n_durations
     in n_durations_range
     for gamma
     in gamma_range
     for beta
     in ast.literal_eval(config[method_header]['beta'])
     for min_kernel_weight
     in ast.literal_eval(config[method_header]['min_kernel_weight'])
     for squared_radius
     in ast.literal_eval(config[method_header]['squared_radius'])
     for n_neighbors
     in n_neighbors_range
     for n_layers
     in ast.literal_eval(config[method_header]['n_layers'])
     for n_nodes
     in ast.literal_eval(config[method_header]['n_nodes'])
     for batch_size
     in ast.literal_eval(config[method_header]['batch_size'])
     for lr
     in lr_range]
if 'sumtune_learning_rate' in config[method_header]:
    sumtune_lr_range = \
        ast.literal_eval(
            config[method_header]['sumtune_learning_rate'])
else:
    sumtune_lr_range = lr_range

hyperparam_hash = hashlib.sha256()
hyperparam_hash.update(str(hyperparams).encode('utf-8'))
hyperparam_hash = hyperparam_hash.hexdigest()

validation_string = 'vr%f' % val_ratio

output_test_table_filename \
    = os.path.join(output_dir,
                   '%s_experiments%d_%s_test_metrics_%s%d.csv'
                   % (estimator_name,
                      n_experiment_repeats,
                      validation_string,
                      hyperparam_hash,
                      finetune_summaries))
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

        if dataset == 'kkbox' and experiment_idx > 0:
            # kkbox takes too much time so we will only run it once
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

        min_time = y_proper_train[:, 0][y_proper_train[:, 1] == 1].min()
        keep_mask = y_proper_train[:, 0] >= min_time
        X_proper_train = X_proper_train[keep_mask]
        y_proper_train = y_proper_train[keep_mask]

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
            best_model_filename = None

            for hyperparam_idx, hyperparam in enumerate(hyperparams):
                pretrain_frac, alpha, sigma, n_durations, gamma, beta, \
                    min_kernel_weight, squared_radius, n_neighbors, n_layers, \
                    n_nodes, batch_size, max_n_epochs, lr = hyperparam

                if alpha == 0 and sigma != sigma_range[0]:
                    continue
                if squared_radius == 0 and gamma > 0:
                    continue

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
                if squared_radius == 0:
                    net = tt.practical.MLPVanilla(
                        X_proper_train_std.shape[1],
                        [n_nodes for layer_idx
                         in range(n_layers)],
                        min(n_nodes,
                            X_proper_train_std.shape[1]),
                        True,
                        0.,
                        output_bias=False)
                else:
                    net = tt.practical.MLPVanilla(
                        X_proper_train_std.shape[1],
                        [n_nodes for layer_idx
                         in range(n_layers)],
                        min(n_nodes,
                            X_proper_train_std.shape[1]),
                        True,
                        0.,
                        output_activation=Hypersphere(
                            squared_radius=squared_radius),
                        output_bias=True)

                model = NKS(net, optimizer,
                            alpha=alpha,
                            sigma=sigma,
                            gamma=gamma,
                            beta=beta,
                            tau=np.sqrt(
                                -np.log(min_kernel_weight)),
                            max_n_neighbors=n_neighbors,
                            dkn_max_n_neighbors=n_neighbors)

                model_filename = \
                    os.path.join(
                        output_dir, 'models',
                        '%s_%s_exp%d_pf%f_a%f_s%f_nd%d_g%f_b%f_mkw%f_sr%f_nn%d_'
                        % (estimator_name, dataset, experiment_idx,
                           pretrain_frac, alpha, sigma, n_durations, gamma,
                           beta, min_kernel_weight, squared_radius, n_neighbors)
                        +
                        'nla%d_nno%d_bs%d_mnep%d_lr%f.pt'
                        % (n_layers, n_nodes, batch_size, max_n_epochs, lr))
                time_elapsed_filename = model_filename[:-3] + '_time.txt'
                epoch_time_elapsed_filename = \
                    model_filename[:-3] + '_epoch_times.txt'
                epoch_times = []

                if n_durations == 0:
                    label_transformer = LabTransDiscreteTime(
                        np.unique(y_proper_train[:, 0][
                            y_proper_train[:, 1] == 1]))
                else:
                    label_transformer = LabTransDiscreteTime(
                        n_durations, scheme='quantiles')
                y_proper_train_discrete = \
                    label_transformer.fit_transform(*y_proper_train.T)
                train_pair = (X_proper_train_std, y_proper_train_discrete)
                train_loader = \
                    model.make_dataloader(train_pair, batch_size, True)
                model.training_data = (X_proper_train_std,
                                       y_proper_train_discrete)
                model.duration_index = label_transformer.cuts
                best_loss = np.inf
                for epoch_idx in range(max_n_epochs):
                    tic_ = time.time()
                    model.fit_dataloader(train_loader, epochs=1)
                    epoch_train_time = time.time() - tic_
                    tic_ = time.time()
                    model.train_embeddings = \
                        model.predict(model.training_data[0],
                                      batch_size=batch_size)
                    model.build_ANN_index()
                    epoch_train_postprocess_time = time.time() - tic_
                    tic_ = time.time()
                    surv_df = \
                        model.interpolate(10).predict_surv_df(X_val_std)
                    y_val_pred = (surv_df.to_numpy(), surv_df.index)
                    val_loss = neg_cindex_td(y_val, y_val_pred, exact=False)
                    epoch_val_time = time.time() - tic_
                    epoch_times.append([epoch_train_time,
                                        epoch_train_postprocess_time,
                                        epoch_val_time])
                    if val_loss != 0.:
                        new_hyperparam = \
                            (pretrain_frac, alpha, sigma, n_durations,
                             gamma, beta, min_kernel_weight, squared_radius,
                             n_neighbors, n_layers, n_nodes, batch_size,
                             epoch_idx + 1, lr, hyperparam_random_seed)
                        print(new_hyperparam,
                              '--',
                              'val loss %f' % val_loss,
                              '--',
                              'train time %f sec(s)'
                              % epoch_train_time,
                              '--',
                              'train postprocess time %f sec(s)'
                              % epoch_train_postprocess_time,
                              '--',
                              'val time %f sec(s)' % epoch_val_time,
                              flush=True)
                        print(new_hyperparam, ':', val_loss, flush=True,
                              file=train_metrics_file)

                        if val_loss < best_loss:
                            best_loss = val_loss
                            wait_idx = 0
                            model.save_net(model_filename)

                            if val_loss < min_loss:
                                min_loss = val_loss
                                arg_min = new_hyperparam
                                best_model_filename = model_filename
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

        # ---------------------------------------------------------------------
        # Using best hyperparameter setting found, set up prediction model and
        # optionally fine-tune summary functions
        #
        X_true_train = X_train[true_train_idx]
        y_true_train = y_train[true_train_idx].astype('float32')

        keep_mask = y_true_train[:, 0] >= min_time
        X_true_train = X_true_train[keep_mask]
        y_true_train = y_true_train[keep_mask]

        X_true_train_std = transform_features(X_true_train, transformer)
        X_true_train_std = X_true_train_std.astype('float32')

        pretrain_frac, alpha, sigma, n_durations, gamma, beta, \
            min_kernel_weight, squared_radius, n_neighbors, n_layers, n_nodes, \
            batch_size, n_epochs, lr, seed = arg_min

        tic = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        optimizer = tt.optim.Adam(lr=lr)
        if squared_radius == 0:
            net = tt.practical.MLPVanilla(X_proper_train_std.shape[1],
                                          [n_nodes for layer_idx
                                           in range(n_layers)],
                                          min(n_nodes,
                                              X_proper_train_std.shape[1]),
                                          True,
                                          0.,
                                          output_bias=False)
        else:
            net = tt.practical.MLPVanilla(X_proper_train_std.shape[1],
                                          [n_nodes for layer_idx
                                           in range(n_layers)],
                                          min(n_nodes,
                                              X_proper_train_std.shape[1]),
                                          True,
                                          0.,
                                          output_activation=Hypersphere(
                                              squared_radius=squared_radius),
                                          output_bias=True)

        model = NKS(net,
                    optimizer,
                    alpha=alpha,
                    sigma=sigma,
                    gamma=gamma,
                    beta=beta,
                    tau=np.sqrt(-np.log(min_kernel_weight)),
                    max_n_neighbors=n_neighbors,
                    dkn_max_n_neighbors=n_neighbors)

        max_n_epochs = int(config[method_header]['max_n_epochs'])
        dataset_max_n_epochs = 'max_n_epochs_%s' % dataset
        if dataset_max_n_epochs in config[method_header]:
            max_n_epochs = \
                int(config[method_header][dataset_max_n_epochs])

        model_filename = \
            os.path.join(
                output_dir, 'models',
                '%s_%s_exp%d_pf%f_a%f_s%f_nd%d_g%f_b%f_mkw%f_sr%f_nn%d_'
                % (estimator_name, dataset, experiment_idx, pretrain_frac,
                   alpha, sigma, n_durations, gamma, beta, min_kernel_weight,
                   squared_radius, n_neighbors)
                +
                'nla%d_nno%d_bs%d_mnep%d_lr%f.pt'
                % (n_layers, n_nodes, batch_size, max_n_epochs, lr))
        time_elapsed_filename = model_filename[:-3] + '_time.txt'

        model.load_net(model_filename)

        # recompute label transformer used for the base neural net
        if n_durations == 0:
            label_transformer = LabTransDiscreteTime(
                np.unique(y_proper_train[:, 0][
                    y_proper_train[:, 1] == 1]))
        else:
            label_transformer = LabTransDiscreteTime(
                n_durations, scheme='quantiles')
        y_proper_train_discrete = \
            label_transformer.fit_transform(*y_proper_train.T)
        model.duration_index = label_transformer.cuts

        y_true_train_discrete = \
            label_transformer.transform(*y_true_train.T)
        model.training_data = (X_true_train_std, y_true_train_discrete)

        model.train_embeddings = \
            model.predict(model.training_data[0], batch_size=batch_size)
        model.build_ANN_index()

        elapsed = float(np.loadtxt(time_elapsed_filename))
        print('Time elapsed (from previous fitting): %f second(s)'
              % elapsed, flush=True)

        if finetune_summaries and beta > 0:
            print('-' * 80)
            print('*** Fine-tuning exemplar summaries ***')
            print()

            # the code is currently written in a janky fashion where the
            # exemplar labels get clobbered when we compute the loss metric
            # on the validation set; we store the initial examplar labels
            # so that we can restore them even after the clobbering
            init_exemplar_labels = model.exemplar_labels.copy()

            summary_finetune_model_filename = model_filename[:-3] + \
                '_summary_finetune.pt'
            summary_finetune_arg_min_filename = \
                summary_finetune_model_filename[:-3] + '_arg_min.txt'
            summary_finetune_min_filename = \
                summary_finetune_model_filename[:-3] + '_min_loss.txt'
            summary_finetune_epoch_time_elapsed_filename = \
                summary_finetune_model_filename[:-3] + '_epoch_times.txt'
            summary_finetune_metrics_filename = \
                summary_finetune_model_filename[:-3] + '_metrics.txt'
            if not os.path.isfile(summary_finetune_model_filename) \
                    or not os.path.isfile(
                        summary_finetune_arg_min_filename) \
                    or not os.path.isfile(
                        summary_finetune_min_filename) \
                    or not os.path.isfile(
                        summary_finetune_epoch_time_elapsed_filename):
                finetune_metrics_file = \
                    open(summary_finetune_metrics_filename, 'w')
                arg_min_finetune = None
                finetune_min_loss = np.inf
                epoch_times = []
                for finetune_lr in sumtune_lr_range:
                    torch.manual_seed(method_random_seed)
                    np.random.seed(method_random_seed)
                    random.seed(method_random_seed)

                    summary_finetune_net = \
                        NKSSummary(model, init_exemplar_labels)
                    summary_model = \
                        Model(summary_finetune_net,
                              NKSSummaryLoss(alpha, sigma),
                              tt.optim.Adam(lr=finetune_lr))
                    train_loader = \
                        summary_model.make_dataloader(
                            (model.train_embeddings,
                             model.training_data[1]),
                            batch_size, True)
                    best_loss = np.inf
                    for epoch_idx in range(max_n_epochs):
                        tic_ = time.time()
                        summary_model.fit_dataloader(train_loader,
                                                     epochs=1)
                        log_exemplar_event_counts = \
                            summary_model.net.log_exemplar_event_counts
                        log_exemplar_censor_counts = \
                            summary_model.net.log_exemplar_censor_counts
                        log_baseline_event_counts = \
                            summary_model.net.log_baseline_event_counts
                        log_baseline_censor_counts = \
                            summary_model.net.log_baseline_censor_counts
                        exemplar_event_counts = \
                            np.exp(log_exemplar_event_counts.detach(
                                ).cpu().numpy())
                        exemplar_censor_counts = \
                            np.exp(log_exemplar_censor_counts.detach(
                                ).cpu().numpy())
                        exemplar_at_risk_counts = \
                            np.flip(
                                np.cumsum(
                                    np.flip(exemplar_event_counts
                                            + exemplar_censor_counts,
                                            axis=1),
                                    axis=1),
                                axis=1)
                        baseline_event_counts = \
                            np.exp(log_baseline_event_counts.detach(
                                ).cpu().numpy())
                        baseline_censor_counts = \
                            np.exp(log_baseline_censor_counts.detach(
                                ).cpu().numpy())
                        baseline_at_risk_counts = \
                            np.flip(
                                np.cumsum(
                                    np.flip(baseline_event_counts
                                            + baseline_censor_counts)))

                        # use `model` rather than `summary_model` since it
                        # has an ANN index already stored, as well as the
                        # interpolation and prediction code
                        model.exemplar_labels[:, 0, :] = \
                            exemplar_event_counts
                        model.exemplar_labels[:, 1, :] = \
                            exemplar_at_risk_counts
                        model.baseline_event_counts = \
                            baseline_event_counts
                        model.baseline_at_risk_counts = \
                            baseline_at_risk_counts
                        epoch_train_time = time.time() - tic_

                        tic_ = time.time()
                        surv_df = \
                            model.interpolate(10).predict_surv_df(
                                X_val_std)
                        y_val_pred = (surv_df.to_numpy(), surv_df.index)
                        val_loss = neg_cindex_td(y_val, y_val_pred,
                                                 exact=False)
                        epoch_val_time = time.time() - tic_
                        epoch_times.append([finetune_lr,
                                            epoch_train_time,
                                            epoch_val_time])
                        if val_loss != 0.:
                            print((epoch_idx + 1, finetune_lr),
                                  '--',
                                  'val loss %f' % val_loss,
                                  '--',
                                  'train time %f sec(s)'
                                  % epoch_train_time,
                                  '--',
                                  'val time %f sec(s)' % epoch_val_time,
                                  flush=True)
                            print(finetune_lr, ':', val_loss, flush=True,
                                  file=finetune_metrics_file)

                            if val_loss < best_loss:
                                best_loss = val_loss
                                wait_idx = 0

                                if val_loss < finetune_min_loss:
                                    finetune_min_loss = val_loss
                                    arg_min_finetune = (epoch_idx + 1,
                                                        finetune_lr)
                                    summary_model.net.ann_index = None
                                    summary_model.save_net(
                                        summary_finetune_model_filename)
                                    summary_model.net.ann_index = \
                                        model.ann_index
                            else:
                                wait_idx += 1
                                if patience > 0 and wait_idx >= patience:
                                    break
                        else:
                            # weird corner case
                            break

                    del summary_model
                    summary_finetune_net = summary_finetune_net.cpu()
                    del summary_finetune_net
                    torch.cuda.empty_cache()

                np.savetxt(summary_finetune_arg_min_filename,
                           np.array(arg_min_finetune).reshape(1, -1))
                np.savetxt(summary_finetune_min_filename,
                           np.array(finetune_min_loss).reshape(1, -1))
                np.savetxt(summary_finetune_epoch_time_elapsed_filename,
                           np.array(epoch_times))

            arg_min_finetune = \
                np.loadtxt(summary_finetune_arg_min_filename).flatten()
            arg_min_finetune_n_epochs, arg_min_finetune_lr = \
                arg_min_finetune
            arg_min_finetune_n_epochs = int(arg_min_finetune_n_epochs)
            finetune_min_loss = \
                float(np.loadtxt(summary_finetune_min_filename))

            torch.manual_seed(method_random_seed)
            np.random.seed(method_random_seed)
            random.seed(method_random_seed)

            if finetune_min_loss < min_loss:
                summary_finetune_net = \
                    NKSSummary(model, init_exemplar_labels)
                summary_model = \
                    Model(summary_finetune_net,
                          NKSSummaryLoss(alpha, sigma),
                          tt.optim.Adam(lr=arg_min_finetune_lr))
                summary_model.load_net(summary_finetune_model_filename)

                log_exemplar_event_counts = \
                    summary_model.net.log_exemplar_event_counts
                log_exemplar_censor_counts = \
                    summary_model.net.log_exemplar_censor_counts
                log_baseline_event_counts = \
                    summary_model.net.log_baseline_event_counts
                log_baseline_censor_counts = \
                    summary_model.net.log_baseline_censor_counts
                exemplar_event_counts = \
                    np.exp(log_exemplar_event_counts.detach(
                        ).cpu().numpy())
                exemplar_censor_counts = \
                    np.exp(log_exemplar_censor_counts.detach(
                        ).cpu().numpy())
                exemplar_at_risk_counts = \
                    np.flip(
                        np.cumsum(
                            np.flip(exemplar_event_counts
                                    + exemplar_censor_counts,
                                    axis=1),
                            axis=1),
                        axis=1)
                baseline_event_counts = \
                    np.exp(log_baseline_event_counts.detach(
                        ).cpu().numpy())
                baseline_censor_counts = \
                    np.exp(log_baseline_censor_counts.detach(
                        ).cpu().numpy())
                baseline_at_risk_counts = \
                    np.flip(
                        np.cumsum(
                            np.flip(baseline_event_counts
                                    + baseline_censor_counts)))
                model.exemplar_labels[:, 0, :] = exemplar_event_counts
                model.exemplar_labels[:, 1, :] = exemplar_at_risk_counts
                model.baseline_event_counts = baseline_event_counts
                model.baseline_at_risk_counts = baseline_at_risk_counts
            else:
                print('*** Warning: Summary fine-tuning did not result'
                      ' in lower validation loss')
                model.exemplar_labels = init_exemplar_labels
                model.baseline_event_counts = None
                model.baseline_at_risk_counts = None

            epoch_times = np.loadtxt(
                summary_finetune_epoch_time_elapsed_filename)
            print('Best summary fine-tuning hyperparameters:',
                  (arg_min_finetune_n_epochs, arg_min_finetune_lr),
                  '-- val loss:', finetune_min_loss)
            print('Time elapsed (from previous fitting): %f second(s)'
                  % epoch_times[:, 1:].sum())

            model.net.to(model.device)  # summary fine-tuning can mess this up

        print()

        # ---------------------------------------------------------------------
        # Test set prediction
        #
        print('Testing...', flush=True)
        X_test_std = transform_features(X_test, transformer)
        X_test_std = X_test_std.astype('float32')
        y_test = y_test.astype('float32')

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
                    '%s_%s_exp%d_pf%f_a%f_s%f_nd%d_g%f_b%f_mkw%f_sr%f_nn%d_'
                    % (estimator_name, dataset, experiment_idx, pretrain_frac,
                       alpha, sigma, n_durations, gamma, beta, min_kernel_weight,
                       squared_radius, n_neighbors)
                    +
                    'nla%d_nno%d_bs%d_nep%d_lr%f_test'
                    % (n_layers, n_nodes, batch_size, n_epochs, lr))
            if finetune_summaries:
                bootstrap_dir = bootstrap_dir[:-5] + '_finetune_summaries_test'
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
