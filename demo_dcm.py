#!/usr/bin/env python
"""
Deep Cox Mixtures baseline (uses the auton-survival PyTorch implementation)

Note that rather than using the auton-survival PyTorch model's `fit` function,
we are using some lower-level functions in the auton-survival package to train
Deep Cox Mixtures; we do this intentionally so that the training is as close as
possible to how we are training all the other models (the other baselines as
well as our proposed survival kernet variants)

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
from torchtuples.tupletree import make_dataloader

from datasets import load_dataset
from metrics import neg_cindex_td

from pycox.models.utils import make_subgrid
from pycox.utils import idx_at_times
from auton_survival.models.dcm.dcm_torch import DeepCoxMixturesTorch
from auton_survival.models.dcm.dcm_utilities import \
    get_hard_z, get_posteriors, get_probability, get_survival, repair_probs, \
    sample_hard_z, smooth_bl_survival, partial_ll_loss
from sksurv.linear_model.coxph import BreslowEstimator


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator_name = 'dcm'

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

hyperparams = \
    [(n_cox_distributions, gamma, smoothing_factor, batch_size, max_n_epochs,
      n_layers, n_nodes, lr)
     for n_cox_distributions
     in ast.literal_eval(config[method_header]['n_cox_distributions'])
     for gamma
     in ast.literal_eval(config[method_header]['gamma'])
     for smoothing_factor
     in ast.literal_eval(config[method_header]['smoothing_factor'])
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


# The `train_step` function is based on the function with the same name in
# `auton_survival.models.dcm.dcm_utilities`; it is being redefined here to work
# with a PyTorch data loader so that, like other models we test, the exact same
# data loader (down to the randomness in batches) is used across models; thus,
# there is no longer a random seed or a batch size in this function definition
# (these are already taken care of in the construction of the data loader)
def train_step(model, dataloader, all_data, breslow_splines,
               optimizer, typ='soft', use_posteriors=True,
               update_splines_after=10, smoothing_factor=1e-4):
    x, (t, e) = all_data
    for i, (xb, (tb, eb)) in enumerate(dataloader):
        # E-Step !!!
        with torch.no_grad():
            posteriors = e_step(model, breslow_splines, xb, tb, eb)

        torch.enable_grad()
        optimizer.zero_grad()
        loss = q_function(model, xb, tb, eb, posteriors, typ)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            try:
                if i % update_splines_after == 0:
                    if use_posteriors:
                        posteriors = []
                        dataloader_no_shuffle = \
                            make_dataloader((x, t, e), 256, False)
                        for xb2, tb2, eb2 in dataloader_no_shuffle:
                            posteriorsb = \
                                e_step(model, breslow_splines, xb2, tb2, eb2)
                            posteriors.append(posteriorsb)
                        posteriors = torch.vstack(posteriors)
                        breslow_splines = \
                            fit_breslow(model, x, t, e, 
                                        posteriors=posteriors,
                                        typ='soft',
                                        smoothing_factor=smoothing_factor)
                    else:
                        breslow_splines = \
                            fit_breslow(model, x, t, e,
                                        posteriors=None,
                                        typ='soft',
                                        smoothing_factor=smoothing_factor)
            except Exception as exce:
                print("Exception!!!:", exce)
                print("Couldn't fit splines, reusing from previous epoch")

    return breslow_splines


# modified to work with setting a device
def e_step(model, breslow_splines, x, t, e):
    if breslow_splines is None:
        # If Breslow splines are not available, like in the first
        # iteration of learning, we randomly compute posteriors.
        posteriors = get_posteriors(torch.rand(len(x), model.k,
                                               device=device))
    else:
        probs = get_likelihood(model, breslow_splines, x, t, e)
        posteriors = get_posteriors(repair_probs(probs))

    return posteriors


# modified to work with setting a device
def q_function(model, x, t, e, posteriors, typ='soft'):

    if typ == 'hard': z = get_hard_z(posteriors)
    else: z = sample_hard_z(posteriors)

    gates, lrisks = model(x)

    k = model.k

    loss = 0
    for i in range(k):
        lrisks_ = lrisks[z == i][:, i]
        loss += partial_ll_loss(lrisks_,
                                t[z == i].cpu().numpy(),
                                e[z == i].cpu().numpy())

    gate_loss = posteriors.exp()*gates
    gate_loss = -torch.sum(gate_loss)
    loss+=gate_loss

    return loss


# modified to work with setting a device
def get_likelihood(model, breslow_splines, x, t, e):
    # Function requires numpy/torch

    gates, lrisks = model(x)
    lrisks = lrisks.cpu().numpy()
    e, t = e.cpu().numpy(), t.cpu().numpy()

    survivals = get_survival(lrisks, breslow_splines, t)
    probability = get_probability(lrisks, breslow_splines, t)

    event_probs = np.array([survivals, probability])
    event_probs = event_probs[e.astype('int'), range(len(e)), :]
    probs = gates + torch.log(torch.tensor(event_probs,
                                           dtype=torch.float32,
                                           device=device))
    return probs


# modified to work with setting a device
def fit_breslow(model, x, t, e, posteriors=None, smoothing_factor=1e-4,
                typ='soft', batch_size=256):
    if posteriors is None:
        dataloader = make_dataloader((x, t, e), batch_size, False)
    else:
        dataloader = make_dataloader((x, t, e, posteriors), batch_size, False)

    lrisks = []
    z = []
    for data in dataloader:
        if posteriors is not None:
            xb, tb, eb, posteriorsb = data
        else:
            xb, tb, eb = data

        gatesb, lrisksb = model(xb)
        lrisksb = lrisksb.cpu().numpy()

        eb = eb.cpu().numpy()
        tb = tb.cpu().numpy()

        if posteriors is None: z_probsb = gatesb
        else: z_probsb = posteriorsb

        if typ == 'soft': zb = sample_hard_z(z_probsb)
        else: zb = get_hard_z(z_probsb)

        lrisks.append(lrisksb)
        z.append(zb)
    lrisks = np.vstack(lrisks)
    z = torch.cat(z).cpu().numpy()

    breslow_splines = {}
    for i in range(model.k):
        breslowk = BreslowEstimator().fit(lrisks[:, i][z==i],
                                          e[z==i].cpu().numpy(),
                                          t[z==i].cpu().numpy())
        breslow_splines[i] = \
            smooth_bl_survival(breslowk,
                               smoothing_factor=smoothing_factor)

    return breslow_splines


# The `predict_survival` function is based on the function with the same name
# in `auton_survival.models.dcm.dcm_utilities`; it is being redefined here to
# work in batches and also to allow for interpolation across time (the argument
# `sub` indicates the subdivision factor for interpolation, where 1 would mean
# no interpolation); we interpolate in the exact same way as how we do it for
# all other models that predict conditional survival functions
def predict_survival(model, x, t, batch_size=256, sub=10):
    if isinstance(t, int) or isinstance(t, float): t = [t]

    model, breslow_splines = model
    dataloader = torch.utils.data.DataLoader(x, batch_size, shuffle=False)

    all_predictions = []
    for xb in dataloader:
        xb = xb.to(device)
        gates, lrisks = model(xb)

        lrisks = lrisks.detach().cpu().numpy()
        gate_probs = torch.exp(gates).detach().cpu().numpy()

        predictions = []
        for t_ in t:
            expert_output = get_survival(lrisks, breslow_splines, t_)
            predictions.append((gate_probs*expert_output).sum(axis=1))

        if sub == 1:  # no interpolation
            all_predictions.append(np.array(predictions).T)
            continue

        s = torch.tensor(np.array(predictions).T)

        # constant PDF interpolation from pycox
        n, m = s.shape
        diff = \
            (s[:, 1:]
             - s[:, :-1]).contiguous().view(-1, 1).repeat(1, sub).view(n, -1)
        rho = torch.linspace(0, 1, sub+1)[:-1].contiguous().repeat(n, m-1)
        s_prev = s[:, :-1].contiguous().view(-1, 1).repeat(1, sub).view(n, -1)
        surv = torch.zeros(n, int((m-1)*sub + 1))
        surv[:, :-1] = diff * rho + s_prev
        surv[:, -1] = s[:, -1]

        all_predictions.append(surv.detach().numpy())

    return np.vstack(all_predictions)


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
        sorted_unique_y_proper_train = \
            np.unique(y_proper_train[:, 0][y_proper_train[:, 1] == 1])

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
            best_model_filename_prefix = 'cache_' + str(uuid.uuid4())
            best_model_filename = best_model_filename_prefix + '.pt'
            best_spline_filename = best_model_filename_prefix + '.pkl'

            for hyperparam_idx, hyperparam in enumerate(hyperparams):
                n_cox_distributions, gamma, smoothing_factor, batch_size, \
                    max_n_epochs, n_layers, n_nodes, lr = hyperparam

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

                n_output_features = min(n_nodes, X_proper_train_std.shape[1])

                net = tt.practical.MLPVanilla(X_proper_train_std.shape[1],
                                              [n_nodes for layer_idx
                                               in range(n_layers)],
                                              n_output_features,
                                              True,
                                              0.,
                                              output_bias=False)
                # note that for `layers`, we just specify the last layer's
                # output dimension; we will set the base neural network to be
                # `net` as defined above
                model = \
                    DeepCoxMixturesTorch(X_proper_train_std.shape[1],
                                         k=n_cox_distributions,
                                         layers=[n_output_features],
                                         gamma=gamma)
                model.embedding = net.to(device)
                model.gate = model.gate.to(device)
                model.expert = model.expert.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                time_elapsed_filename = \
                    os.path.join(output_dir, 'models',
                                 '%s_%s_exp%d_'
                                 % (estimator_name, dataset, experiment_idx)
                                 +
                                 'k%d_g%f_sf%f_'
                                 % (n_cox_distributions, gamma,
                                    smoothing_factor)
                                 +
                                 'bs%d_mnep%d_nla%d_nno%d_'
                                 % (batch_size, max_n_epochs, n_layers,
                                    n_nodes)
                                 +
                                 'lr%f_time.txt'
                                 % lr)
                epoch_time_elapsed_filename = \
                    time_elapsed_filename[:-8] + 'epoch_times.txt'
                epoch_times = []

                train_pair = (torch.tensor(X_proper_train_std,
                                           dtype=torch.float32,
                                           device=device),
                              (torch.tensor(y_proper_train[:, 0],
                                            dtype=torch.float32,
                                            device=device),
                               torch.tensor(y_proper_train[:, 1],
                                            dtype=torch.float32,
                                            device=device)))
                train_loader = make_dataloader(train_pair, batch_size, True)
                best_loss = np.inf
                breslow_splines = None
                for epoch_idx in range(max_n_epochs):
                    tic_ = time.time()
                    try:
                        breslow_splines = \
                            train_step(model,
                                       train_loader,
                                       train_pair,
                                       breslow_splines,
                                       optimizer,
                                       smoothing_factor=smoothing_factor)
                    except Exception as exc:
                        print('*** Exception:', exc)
                        print('*** Warning: Terminating training early')
                        break
                    epoch_train_time = time.time() - tic_
                    tic_ = time.time()
                    model.eval()
                    surv = predict_survival((model, breslow_splines),
                                            X_val_std,
                                            sorted_unique_y_proper_train,
                                            sub=10)
                    model.train()
                    interpolated_time_grid = \
                        make_subgrid(sorted_unique_y_proper_train, 10)
                    interpolated_time_grid = \
                        np.array([t for t in interpolated_time_grid],
                                 dtype=np.float32)
                    y_val_pred = (surv.T, interpolated_time_grid)
                    val_loss = neg_cindex_td(y_val, y_val_pred, exact=False)
                    epoch_val_time = time.time() - tic_
                    epoch_times.append([epoch_train_time,
                                        epoch_val_time])
                    if val_loss != 0.:
                        new_hyperparam = \
                            (n_cox_distributions, gamma, smoothing_factor,
                             batch_size, epoch_idx + 1, n_layers,
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
                                torch.save(model.state_dict(),
                                           best_model_filename)
                                with open(best_spline_filename, 'wb') as f:
                                    pickle.dump(breslow_splines, f)
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

            n_cox_distributions, gamma, smoothing_factor, batch_size, \
                n_epochs, n_layers, n_nodes, lr, seed = arg_min
            model_filename = \
                os.path.join(output_dir, 'models',
                             '%s_%s_exp%d_'
                             % (estimator_name, dataset, experiment_idx)
                             +
                             'k%d_g%f_sf%f_'
                             % (n_cox_distributions, gamma, smoothing_factor)
                             +
                             'bs%d_nep%d_nla%d_nno%d_lr%f_test.pt'
                             % (batch_size, n_epochs, n_layers, n_nodes, lr))
            spline_filename = model_filename[:-2] + 'pkl'
            time_elapsed_filename = \
                os.path.join(output_dir, 'models',
                             '%s_%s_exp%d_'
                             % (estimator_name, dataset, experiment_idx)
                             +
                             'k%d_g%f_sf%f_'
                             % (n_cox_distributions, gamma, smoothing_factor)
                             +
                             'bs%d_mnep%d_nla%d_nno%d_'
                             % (batch_size, max_n_epochs, n_layers,
                                n_nodes)
                             +
                             'lr%f_time.txt'
                             % lr)
            os.rename(best_model_filename, model_filename)
            os.rename(best_spline_filename, spline_filename)
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

        n_cox_distributions, gamma, smoothing_factor, batch_size, n_epochs, \
            n_layers, n_nodes, lr, seed = arg_min

        tic = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        n_output_features = min(n_nodes, X_proper_train_std.shape[1])

        net = tt.practical.MLPVanilla(X_proper_train_std.shape[1],
                                      [n_nodes for layer_idx
                                       in range(n_layers)],
                                      n_output_features,
                                      True,
                                      0.,
                                      output_bias=False)
        # note that for `layers`, we just specify the last layer's
        # output dimension; we will set the base neural network to be
        # `net` as defined above
        model = \
            DeepCoxMixturesTorch(X_proper_train_std.shape[1],
                                 k=n_cox_distributions,
                                 layers=[n_output_features],
                                 gamma=gamma)
        model.embedding = net.to(device)
        model.gate = model.gate.to(device)
        model.expert = model.expert.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model_filename = \
            os.path.join(output_dir, 'models',
                         '%s_%s_exp%d_'
                         % (estimator_name, dataset, experiment_idx)
                         +
                         'k%d_g%f_sf%f_'
                         % (n_cox_distributions, gamma, smoothing_factor)
                         +
                         'bs%d_nep%d_nla%d_nno%d_lr%f_test.pt'
                         % (batch_size, n_epochs, n_layers, n_nodes, lr))
        spline_filename = model_filename[:-2] + 'pkl'
        time_elapsed_filename = model_filename[:-3] + '_time.txt'
        model.load_state_dict(torch.load(model_filename))
        model.eval()
        with open(spline_filename, 'rb') as f:
            breslow_splines = pickle.load(f)
        elapsed = float(np.loadtxt(time_elapsed_filename))
        print('Time elapsed (from previous fitting): %f second(s)'
              % elapsed, flush=True)

        surv = predict_survival((model, breslow_splines),
                                X_test_std,
                                sorted_unique_y_proper_train,
                                sub=10)
        interpolated_time_grid = \
            make_subgrid(sorted_unique_y_proper_train, 10)
        interpolated_time_grid = \
            np.array([t for t in interpolated_time_grid],
                     dtype=np.float32)
        y_test_pred = (surv.T, interpolated_time_grid)

        loss = neg_cindex_td(y_test, y_test_pred)
        print('Hyperparameter', arg_min, 'achieves test loss %f' % loss,
              flush=True)

        test_set_metrics = [loss]

        if not compute_bootstrap_CI:
            final_test_scores[arg_min] = tuple(test_set_metrics)
        else:
            rng = np.random.RandomState(bootstrap_random_seed)

            bootstrap_dir = \
                os.path.join(output_dir, 'bootstrap',
                             '%s_%s_exp%d_'
                             % (estimator_name, dataset, experiment_idx)
                             +
                             'k%d_g%f_sf%f_'
                             % (n_cox_distributions, gamma, smoothing_factor)
                             +
                             'bs%d_nep%d_nla%d_nno%d_lr%f_test'
                             % (batch_size, n_epochs, n_layers, n_nodes, lr))
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
