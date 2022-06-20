"""
Some helper functions for visualization

Author: George H. Chen (georgechen [at symbol] cmu.edu)
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from matplotlib import colors
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder


def compute_SFT_survival_curves(exemplar_labels, baseline_event_counts,
                                baseline_at_risk_counts, eps=1e-7):
    hazards = (exemplar_labels[:, 0, :] + baseline_event_counts) / \
        (exemplar_labels[:, 1, :] + baseline_at_risk_counts)
    return np.exp(np.cumsum(np.log(1 - hazards + eps), axis=1))


def compute_median_survival_times(survival_curves, times):
    median_survival_times = []
    for curve in survival_curves:
        # use same code logic as lifelines
        if curve[-1] > .5:
            median_survival_times.append(np.inf)
        else:
            median_survival_times.append(
                times[np.searchsorted(-curve, [-.5])[0]])
    return np.array(median_survival_times)


def transform(dataset, X, continuous_n_bins=5):
    if dataset == 'support':
        feature_names = \
            ['age', 'female', 'race', 'number of comorbidities', 'diabetes',
             'dementia', 'cancer', 'mean arterial blood pressure',
             'heart rate', 'respiration rate', 'temperature',
             'white blood count', 'serum sodium', 'serum creatinine']

        # binary: 1, 4, 5
        # categorical: 2, 6
        # continuous: 0, 3, 7, 8, 9, 10, 11, 12, 13
        binary_indices = [1, 4, 5]
        continuous_indices = [0, 3, 7, 8, 9, 10, 11, 12, 13]
        discretized_features = []
        discretized_feature_names = []
        for idx in continuous_indices:
            n_bins_to_use = continuous_n_bins
            discretizer = KBinsDiscretizer(n_bins=n_bins_to_use,
                                           strategy='quantile',
                                           encode='onehot-dense')
            new_features = discretizer.fit_transform(
                X[:, idx].reshape(-1, 1).astype(float))
            if discretizer.n_bins_[0] != n_bins_to_use:
                n_bins_to_use = discretizer.n_bins_[0]
            if n_bins_to_use > 1:
                discretized_features.append(new_features)
                for bin_idx in range(n_bins_to_use):
                    if bin_idx == 0:
                        discretized_feature_names.append(
                            feature_names[idx]
                            + ' bin#1(-inf,%.2f)'
                            % discretizer.bin_edges_[0][bin_idx+1])
                    elif bin_idx == n_bins_to_use - 1:
                        discretized_feature_names.append(
                            feature_names[idx]
                            + ' bin#%d[%.2f,inf)'
                            % (n_bins_to_use,
                               discretizer.bin_edges_[0][bin_idx]))
                    else:
                        discretized_feature_names.append(
                            feature_names[idx]
                            + ' bin#%d[%.2f,%.2f)'
                            % tuple([bin_idx + 1] +
                                    list(discretizer.bin_edges_[0][
                                        bin_idx:bin_idx+2])))
        for idx in binary_indices:
            discretized_features.append(
                X[:, idx].reshape(-1, 1).astype(float))
            discretized_feature_names.append(feature_names[idx])

        # race
        discretizer = OneHotEncoder(sparse=False,
                                    categories=[[0, 1, 2, 3, 4, 5]])
        discretized_features.append(
            discretizer.fit_transform(
                X[:, 2].reshape(-1, 1).astype(float)))
        discretized_feature_names.extend(['race cat#1(unspecified)',
                                          'race cat#2(asian)',
                                          'race cat#3(black)',
                                          'race cat#4(hispanic)',
                                          'race cat#5(other)',
                                          'race cat#6(white)'])

        # cancer
        discretizer = OneHotEncoder(sparse=False,
                                    categories=[[0, 1, 2]])
        discretized_features.append(
            discretizer.fit_transform(
                X[:, 6].reshape(-1, 1).astype(float)))
        discretized_feature_names.extend(['cancer cat#1(no)',
                                          'cancer cat#2(yes)',
                                          'cancer cat#3(metastatic)'])
        return np.hstack(discretized_features), discretized_feature_names

    elif dataset == 'kkbox':
        feature_names = [
            'log_days_between_subs', 'n_prev_churns',
            'log_days_since_reg_init', 'log_payment_plan_days',
            'log_plan_list_price', 'log_actual_amount_paid', 'age_at_start',
            'gender', 'city', 'registered_via', 'no_prev_churns', 'is_cancel',
            'strange_age', 'is_auto_renew', 'nan_days_since_reg_init']
        continuous_indices = [0, 1, 2, 3, 4, 5, 6]
        categorical_indices = [7, 8, 9]
        leave_indices = [10, 11, 12, 13, 14]
        discretized_features = []
        discretized_feature_names = []
        for idx in continuous_indices:
            n_bins_to_use = continuous_n_bins
            discretizer = KBinsDiscretizer(n_bins=n_bins_to_use,
                                           strategy='quantile',
                                           encode='onehot-dense')
            new_features = discretizer.fit_transform(
                X[:, idx].reshape(-1, 1).astype(float))
            if discretizer.n_bins_[0] != n_bins_to_use:
                n_bins_to_use = discretizer.n_bins_[0]
            if n_bins_to_use > 1:
                discretized_features.append(new_features)
                for bin_idx in range(n_bins_to_use):
                    if bin_idx == 0:
                        discretized_feature_names.append(
                            feature_names[idx]
                            + ' bin#1(-inf,%.2f)'
                            % discretizer.bin_edges_[0][bin_idx+1])
                    elif bin_idx == n_bins_to_use - 1:
                        discretized_feature_names.append(
                            feature_names[idx]
                            + ' bin#%d[%.2f,inf)'
                            % (n_bins_to_use,
                               discretizer.bin_edges_[0][bin_idx]))
                    else:
                        discretized_feature_names.append(
                            feature_names[idx]
                            + ' bin#%d[%.2f,%.2f)'
                            % tuple([bin_idx + 1] +
                                    list(discretizer.bin_edges_[0][
                                        bin_idx:bin_idx+2])))

        for idx in categorical_indices:
            feature_name = feature_names[idx]
            discretizer = OneHotEncoder(sparse=False)
            discretized_features.append(
                discretizer.fit_transform(X[:, idx].reshape(-1, 1)))
            for cat_idx, cat in enumerate(discretizer.categories_[0]):
                discretized_feature_names.append(
                    feature_name + ' cat#%d(%s)' % (cat_idx + 1, cat))

        for idx in leave_indices:
            discretized_features.append(
                X[:, idx].reshape(-1, 1).astype(float))
            discretized_feature_names.append(feature_names[idx])

        return np.hstack(discretized_features), discretized_feature_names

    else:
        raise NotImplementedError


def argsort_feature_names(
        feature_names, heatmap, score_func, aggregate_func,
        reverse=False, max_features_to_display=None, verbose=False):
    score_dict = dict()
    for v_i, v in enumerate(feature_names):
        if "bin#" in v:
            base_feature = v[:(v.index('bin#')-1)]
        elif "cat#" in v:
            base_feature = v[:(v.index('cat#')-1)]
        else:
            base_feature = v

        score = score_func(heatmap[v_i])

        if base_feature in score_dict:
            score_dict[base_feature] = aggregate_func(score_dict[base_feature],
                                                      score)
        else:
            score_dict[base_feature] = score

    sorted_key_val = sorted(score_dict.items(), key=lambda pair: pair[1],
                            reverse=reverse)

    new_order = []
    if verbose:
        print('Top scores (across row):')
    for curr_base, score in sorted_key_val:
        if verbose:
            print(curr_base, ':', score)
        block_to_add = []
        for v_i, v in enumerate(feature_names):
            if "bin#" in v:
                base_feature = v[:(v.index('bin#')-1)]
            elif "cat#" in v:
                base_feature = v[:(v.index('cat#')-1)]
            else:
                base_feature = v

            if curr_base == base_feature:
                block_to_add.append(v_i)
        if max_features_to_display is not None and \
                len(new_order) + len(block_to_add) > max_features_to_display:
            print()
            print(len(block_to_add))
            print([feature_names[_] for _ in block_to_add])
            print()
            break
        new_order += block_to_add
    new_order = np.array(new_order)
    return new_order


modified_red_color_map = \
    colors.LinearSegmentedColormap(
        'modified_red_color_map',
        segmentdata={
            'red':   ((0.0,  1.0, 1.0),
                      (0.9,  0.9, 0.9),
                      (1.0,  0.6, 0.6)),
            'green': ((0.0,  1.0, 1.0),
                      (1.0,  0.0, 0.0)),
            'blue':  ((0.0,  1.0, 1.0),
                      (1.0,  0.0, 0.0))},
        N=256)

modified_gray_color_map = \
    colors.LinearSegmentedColormap(
        'modified_gray_color_map',
        segmentdata={
            'red':   ((0.0,  1.0, 1.0),
                      (0.2,  0.9, 0.9),
                      (1.0,  0., 0.)),
            'green': ((0.0,  1.0, 1.0),
                      (0.2,  0.9, 0.9),
                      (1.0,  0., 0.)),
            'blue':  ((0.0,  1.0, 1.0),
                      (0.2,  0.9, 0.9),
                      (1.0,  0., 0.))},
        N=256)


def heatmap_plot(heatmap, median_survival_times, feature_names,
                 max_features_to_display=None, max_observed_times=None,
                 save_filename=None, show_plot=True, units=None,
                 custom_xlabel=None):
    feature_names = np.array(feature_names).copy()
    heatmap = heatmap.copy()

    # sort by max probability first (and then we do another sort)
    new_order = argsort_feature_names(feature_names, heatmap, np.max,
                                      max, reverse=True)
    feature_names = feature_names[new_order]
    heatmap = heatmap[new_order]

    # sort by peak-to-peak
    new_order = argsort_feature_names(
        feature_names, heatmap, np.ptp, max, reverse=True,
        max_features_to_display=max_features_to_display)
    feature_names = feature_names[new_order]
    heatmap = heatmap[new_order]

    for idx, f in enumerate(feature_names):
        if 'bin#1(-inf' in f:
            idx2 = f.index('bin#1(-inf')
            feature_names[idx] = f[:idx2] + '< ' + f[idx2+11:-1]
        elif 'bin#' in f and 'inf)' in f:
            idx2 = f.index('bin#')
            feature_names[idx] = f[:idx2] + '≥ ' + f[idx2+6:-1].split(',')[0]
        elif 'bin#' in f:
            idx2 = f.index('bin#')
            feature_names[idx] = f.replace(f[idx2:idx2+5], 'in ')
        elif 'cat#' in f:
            idx2 = f.index('cat#')
            idx3 = f[idx2+4:].index('(')
            feature_names[idx] = f[:idx2] + '= ' + f[idx2+5+idx3:-1]

    with plt.style.context('seaborn-dark'):
        fig = plt.figure(figsize=(len(median_survival_times)*1.0,
                                  len(feature_names)/3*0.7))
        ax = fig.gca()
        ax.xaxis.tick_top()
        ax.set_xlabel('X LABEL')
        ax.xaxis.set_label_position('top')
        # sb.heatmap(heatmap, cmap=modified_red_color_map)
        sb.heatmap(heatmap, cmap=modified_gray_color_map)
        cbar = ax.collections[0].colorbar

        # ticklabels = cbar.ax.get_ymajorticklabels()
        ticks = list(cbar.get_ticks())
        cbar.set_ticks([0., heatmap.max()] + ticks)
        # cbar.set_ticklabels([0., heatmap.max()] + ticklabels)

        plt.ylabel('Feature')
        if custom_xlabel is None:
            if units is None:
                plt.xlabel('Clusters sorted by median survival time')
            else:
                plt.xlabel('Clusters sorted by median survival time (%s)'
                           % units)
        else:
            plt.xlabel(custom_xlabel)
        plt.yticks(np.arange(len(feature_names)) + 0.5,
                   list(feature_names),
                   rotation='horizontal')
        if max_observed_times is not None:
            plt.xticks(np.arange(len(median_survival_times)) + 0.5,
                       [np.isinf(median_time) and (" >%.2f" % max_time)
                        or " %.2f " % median_time
                        for median_time, max_time
                        in zip(median_survival_times, max_observed_times)])
        if save_filename is not None:
            plt.savefig(save_filename, bbox_inches='tight')
        if not show_plot:
            plt.close()
