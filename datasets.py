"""
Code for loading different datasets in a unified format

Author: George H. Chen (georgechen [at symbol] cmu.edu)
"""
#!/usr/bin/env python
import csv
import os
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import scipy
import scipy.io
import statsmodels.api as sm
import torch
from lifelines.datasets import load_regression_dataset
from pycox.datasets import flchain, support, kkbox
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision import datasets as visiondatasets


def load_dataset(dataset, random_seed_offset=0, fix_test_shuffle_train=False):
    """
    Loads a dataset.

    Parameters
    ----------
    dataset : string
        One of 'rotterdam-gbsg', 'support', or 'kkbox'. For simplicity, we have
        removed support for the UNOS dataset in our public code release as the
        UNOS dataset requires special access.

    random_seed_offset : int, optional (default=0)
        Offset to add to random seed in shuffling the data.

    Returns
    -------
    X_train : 2D numpy array, shape = [n_samples, n_features]
        Training feature vectors.

    y_train : numpy array, shape = [n_samples, 2]
        Training labels. The first column is for observed times and the second
        column is for event indicators. The i-th row corresponds to the i-th
        row in `X_train`.

    X_test : 2D numpy array
        Test feature vectors.

    y_test : numpy array (2D for survival analysis)
        Test labels.

    feature_names : list
        List of strings specifying the names of the features (columns of
        `X_train` and `X_test`).

    compute_features_and_transformer : function
        Function for fitting and then transforming features into some
        "standardized"/"normalized" feature space. This should be applied to
        training feature vectors prior to using a learning algorithm (unless the
        learning algorithm does not need this sort of normalization). This
        function returns both the normalized features and a transformer object
        (see the next output for how to use this transformer object).

    transform_features : function
        Function that, given feature vectors (e.g., validation/test data) and a
        transformer object (created via `compute_features_and_transformer`),
        transforms the feature vectors into a normalized feature space.

    fixed_train_test_split : boolean
        If True, then this means that the dataset comes with its own train/test
        split and therefore there is no randomization for this split.
    """
    fixed_train_test_split = False
    test_size = 0.3
    train_dataset = None
    test_dataset = None
    if dataset == 'support':
        with open('data/support2.csv', 'r') as f:
            csv_reader = csv.reader(f)
            header = True
            X = []
            y = []
            for row in csv_reader:
                if header:
                    header = False
                else:
                    row = row[1:]

                    age = float(row[0])
                    sex = int(row[2] == 'female')

                    race = row[16]
                    if race == '':
                        race = 0
                    elif race == 'asian':
                        race = 1
                    elif race == 'black':
                        race = 2
                    elif race == 'hispanic':
                        race = 3
                    elif race == 'other':
                        race = 4
                    elif race == 'white':
                        race = 5

                    num_co = int(row[8])
                    diabetes = int(row[22])
                    dementia = int(row[23])

                    ca = row[24]
                    if ca == 'no':
                        ca = 0
                    elif ca == 'yes':
                        ca = 1
                    elif ca == 'metastatic':
                        ca = 2

                    meanbp = row[29]
                    if meanbp == '':
                        meanbp = np.nan
                    else:
                        meanbp = float(meanbp)

                    hrt = row[31]
                    if hrt == '':
                        hrt = np.nan
                    else:
                        hrt = float(hrt)

                    resp = row[32]
                    if resp == '':
                        resp = np.nan
                    else:
                        resp = float(resp)

                    temp = row[33]
                    if temp == '':
                        temp = np.nan
                    else:
                        temp = float(temp)

                    wblc = row[30]
                    if wblc == '':
                        wblc = np.nan
                    else:
                        wblc = float(wblc)

                    sod = row[38]
                    if sod == '':
                        sod = np.nan
                    else:
                        sod = float(sod)

                    crea = row[37]
                    if crea == '':
                        crea = np.nan
                    else:
                        crea = float(crea)

                    d_time = float(row[5])
                    death = int(row[1])

                    X.append([age, sex, race, num_co, diabetes, dementia, ca,
                              meanbp, hrt, resp, temp, wblc, sod, crea])
                    y.append([d_time, death])

        X = np.array(X)
        y = np.array(y)

        not_nan_mask = ~np.isnan(X).any(axis=1)
        X = X[not_nan_mask]
        y = y[not_nan_mask]

        feature_names = ['age', 'sex', 'num.co', 'diabetes', 'dementia', 'ca',
                         'meanbp', 'hrt', 'resp', 'temp', 'wblc', 'sod', 'crea',
                         'race_blank', 'race_asian', 'race_black',
                         'race_hispanic', 'race_other', 'race_white']

        categories = [list(range(int(X[:, 2].max()) + 1))]

        def compute_features_and_transformer(features, cox=False):
            new_features = np.zeros((features.shape[0], 19))
            scaler = StandardScaler()
            encoder = OneHotEncoder(categories=categories)
            cols_standardize = [0, 7, 8, 9, 10, 11, 12, 13]
            cols_leave = [1, 4, 5]
            cols_categorical = [2]
            new_features[:, [0, 6, 7, 8, 9, 10, 11, 12]] = \
                scaler.fit_transform(features[:, cols_standardize])
            new_features[:, [1, 3, 4]] = features[:, cols_leave]
            new_features[:, 13:] = \
                encoder.fit_transform(features[:, cols_categorical]).toarray()
            new_features[:, 2] = features[:, 3] / 9.
            new_features[:, 5] = features[:, 6] / 2.
            if cox:
                return new_features[:, :-1], (scaler, encoder)
            return new_features, (scaler, encoder)

        def transform_features(features, transformer, cox=False):
            new_features = np.zeros((features.shape[0], 19))
            scaler, encoder = transformer
            cols_standardize = [0, 7, 8, 9, 10, 11, 12, 13]
            cols_leave = [1, 4, 5]
            cols_categorical = [2]
            new_features[:, [0, 6, 7, 8, 9, 10, 11, 12]] = \
                scaler.transform(features[:, cols_standardize])
            new_features[:, [1, 3, 4]] = features[:, cols_leave]
            new_features[:, 13:] = \
                encoder.transform(features[:, cols_categorical]).toarray()
            new_features[:, 2] = features[:, 3] / 9.
            new_features[:, 5] = features[:, 6] / 2.
            if cox:
                return new_features[:, :-1]
            return new_features

        dataset_random_seed = 331231101

    elif dataset == 'rotterdam-gbsg':
        # ----------------------------------------------------------------------
        # snippet of code from DeepSurv repository
        datasets = defaultdict(dict)
        with h5py.File('data/gbsg_cancer_train_test.h5', 'r') as fp:
            for ds in fp:
                for array in fp[ds]:
                    datasets[ds][array] = fp[ds][array][:]
        # ----------------------------------------------------------------------

        feature_names = ['horTh', 'tsize', 'menostat', 'age', 'pnodes',
                         'progrec', 'estrec']

        X_train = datasets['train']['x']
        y_train = np.array([datasets['train']['t'], datasets['train']['e']]).T
        X_test = datasets['test']['x']
        y_test = np.array([datasets['test']['t'], datasets['test']['e']]).T

        def compute_features_and_transformer(features, cox=False):
            new_features = np.zeros_like(features)
            transformer = StandardScaler()
            cols_standardize = [3, 4, 5, 6]
            cols_leave = [0, 2]
            new_features[:, cols_standardize] = \
                transformer.fit_transform(features[:, cols_standardize])
            new_features[:, cols_leave] = features[:, cols_leave]
            new_features[:, 1] = features[:, 1] / 2.
            return new_features, transformer

        def transform_features(features, transformer, cox=False):
            new_features = np.zeros_like(features)
            cols_standardize = [3, 4, 5, 6]
            cols_leave = [0, 2]
            new_features[:, cols_standardize] = \
                transformer.transform(features[:, cols_standardize])
            new_features[:, cols_leave] = features[:, cols_leave]
            new_features[:, 1] = features[:, 1] / 2.
            return new_features

        dataset_random_seed = 1831262265
        fixed_train_test_split = True

    elif dataset == 'kkbox':
        df = kkbox.read_df()
        if df is None:
            kkbox.download_kkbox()
            df = kkbox.read_df()

        feature_names = ['log_days_between_subs', 'n_prev_churns',
                'log_days_since_reg_init', 'log_payment_plan_days',
                'log_plan_list_price', 'log_actual_amount_paid',
                'age_at_start', 'gender', 'city', 'registered_via',
                'no_prev_churns', 'is_cancel', 'strange_age',
                'is_auto_renew', 'nan_days_since_reg_init']
        X = df[['log_days_between_subs', 'n_prev_churns',
                'log_days_since_reg_init', 'log_payment_plan_days',
                'log_plan_list_price', 'log_actual_amount_paid',
                'age_at_start', 'gender', 'city', 'registered_via',
                'no_prev_churns', 'is_cancel', 'strange_age',
                'is_auto_renew', 'nan_days_since_reg_init']].to_numpy()
        y = df[['duration', 'event']].to_numpy()
        X[:, -5:] = X[:, -5:].astype(np.float)
        X[:, 7][df['gender'].isna().to_numpy()] = 'n/a'
        X[:, 8][df['city'].isna().to_numpy()] = 0
        X[:, 9][df['registered_via'].isna().to_numpy()] = 0
        X[:, 8:10] = X[:, 8:10].astype(np.int)

        categories = \
            [np.array(['female', 'male', 'n/a'], dtype=np.object),
             np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                       17, 18, 19, 20, 21, 22], dtype=np.object),
             np. array([0, 3, 4, 7, 9, 10, 13, 16], dtype=np.object)]

        def compute_features_and_transformer(features, cox=False):
            new_features = np.zeros((features.shape[0], 45), dtype=np.float)
            scaler = StandardScaler()
            encoder = OneHotEncoder(categories=categories)
            cols_standardize = [0, 1, 2, 3, 4, 5, 6]
            cols_categorical = [7, 8, 9]
            cols_leave = [10, 11, 12, 13, 14]
            one_hot_representation = \
                encoder.fit_transform(features[:, cols_categorical]).toarray()
            new_features[:, cols_standardize] = \
                scaler.fit_transform(features[:, cols_standardize])
            new_features[:, 7:40] = \
                encoder.fit_transform(features[:, cols_categorical]).toarray()
            new_features[:, 40:] = features[:, cols_leave]
            if cox:
                # drop 9, 31, 39 corresponding to last categories of each
                # categorical variable
                return new_features[:, [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                        30, 32, 33, 34, 35, 36, 37, 38,
                                        40, 41, 42, 43, 44]], (scaler, encoder)
            return new_features, (scaler, encoder)

        def transform_features(features, transformer, cox=False):
            new_features = np.zeros((features.shape[0], 45), dtype=np.float)
            scaler, encoder = transformer
            num_one_hot_features = np.sum([len(x) for x in encoder.categories_])
            cols_standardize = [0, 1, 2, 3, 4, 5, 6]
            cols_categorical = [7, 8, 9]
            cols_leave = [10, 11, 12, 13, 14]
            new_features[:, cols_standardize] = \
                scaler.transform(features[:, cols_standardize])
            new_features[:, 7:40] = \
                encoder.transform(features[:, cols_categorical]).toarray()
            new_features[:, 40:] = features[:, cols_leave]
            if cox:
                # drop 9, 31, 39 corresponding to last categories of each
                # categorical variable
                return new_features[:, [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                        30, 32, 33, 34, 35, 36, 37, 38,
                                        40, 41, 42, 43, 44]]
            return new_features

        dataset_random_seed = 172428017

    else:
        raise NotImplementedError('Unsupported dataset: %s' % dataset)

    if fixed_train_test_split:
        # shuffle the training data but leave the test data as is
        if fix_test_shuffle_train:
            rng = np.random.RandomState(dataset_random_seed
                                        + random_seed_offset)
        else:
            rng = np.random.RandomState(dataset_random_seed)
        shuffled_indices = rng.permutation(len(X_train))
        X_train = X_train[shuffled_indices]
        y_train = y_train[shuffled_indices]
    else:
        # note that by default, sklearn's `train_test_split` will shuffle the
        # data before doing the split
        if fix_test_shuffle_train:
            rng = np.random.RandomState(dataset_random_seed)
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=test_size, random_state=rng)
            if random_seed_offset > 0:
                rng = np.random.RandomState(dataset_random_seed
                                            + random_seed_offset)
                shuffled_indices = rng.permutation(len(X_train))
                X_train = X_train[shuffled_indices]
                y_train = y_train[shuffled_indices]
        else:
            rng = np.random.RandomState(dataset_random_seed
                                        + random_seed_offset)
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=test_size, random_state=rng)

    return X_train, y_train, X_test, y_test, feature_names, \
            compute_features_and_transformer, transform_features, \
            fixed_train_test_split
