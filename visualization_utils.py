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
    if dataset == 'rotterdam-gbsg':
        feature_names = ['hormonal therapy',
                         'tumor size',
                         'postmenopausal',
                         'age',
                         'number of positive nodes',
                         'progesterone receptor',
                         'estrogen receptor']
        leave_indices = [0, 2]
        continuous_indices = [1, 3, 4, 5, 6]
        discretized_features = []
        discretized_feature_names = []
        all_n_bins_to_use = []
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
            else:
                raise Exception('Single discretization bin encountered for '
                                + 'feature "%s"' % feature_names[idx])
            all_n_bins_to_use.append(n_bins_to_use)

        for idx in leave_indices:
            discretized_features.append(X[:, idx].reshape(-1, 1).astype(float))
            discretized_feature_names.append(feature_names[idx])
            all_n_bins_to_use.append(1)

        return np.hstack(discretized_features), discretized_feature_names, \
            all_n_bins_to_use

    elif dataset == 'support':
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
        all_n_bins_to_use = []
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
            else:
                raise Exception('Single discretization bin encountered for '
                                + 'feature "%s"' % feature_names[idx])
            all_n_bins_to_use.append(n_bins_to_use)
        for idx in binary_indices:
            discretized_features.append(
                X[:, idx].reshape(-1, 1).astype(float))
            discretized_feature_names.append(feature_names[idx])
            all_n_bins_to_use.append(1)

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
        all_n_bins_to_use.append(6)

        # cancer
        discretizer = OneHotEncoder(sparse=False,
                                    categories=[[0, 1, 2]])
        discretized_features.append(
            discretizer.fit_transform(
                X[:, 6].reshape(-1, 1).astype(float)))
        discretized_feature_names.extend(['cancer cat#1(no)',
                                          'cancer cat#2(yes)',
                                          'cancer cat#3(metastatic)'])
        all_n_bins_to_use.append(3)
        return np.hstack(discretized_features), discretized_feature_names, \
            all_n_bins_to_use

    elif dataset == 'kkbox':
        feature_names = [
            'log_days_between_subs', 'n_prev_churns',
            'log_days_since_reg_init', 'log_payment_plan_days',
            'log_plan_list_price', 'log_actual_amount_paid', 'age_at_start',
            'gender', 'city', 'registered_via', 'no_prev_churns', 'is_cancel',
            'strange_age', 'is_auto_renew', 'nan_days_since_reg_init']
        binarize_indices = [0, 1]
        continuous_indices = [2, 3, 4, 5, 6]
        categorical_indices = [7, 8, 9]
        leave_indices = [10, 11, 12, 13, 14]
        discretized_features = []
        discretized_feature_names = []
        all_n_bins_to_use = []

        for idx in binarize_indices:
            discretized_features.append(
                (1 * (X[:, idx] > 0)).reshape(-1, 1).astype(float))
            discretized_feature_names.append(feature_names[idx] + ' > 0')
            all_n_bins_to_use.append(1)

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
            else:
                raise Exception('Single discretization bin encountered for '
                                + 'feature "%s"' % feature_names[idx])
            all_n_bins_to_use.append(n_bins_to_use)

        for idx in categorical_indices:
            feature_name = feature_names[idx]
            discretizer = OneHotEncoder(sparse=False)
            discretized_features.append(
                discretizer.fit_transform(X[:, idx].reshape(-1, 1)))
            for cat_idx, cat in enumerate(discretizer.categories_[0]):
                discretized_feature_names.append(
                    feature_name + ' cat#%d(%s)' % (cat_idx + 1, cat))
            all_n_bins_to_use.append(len(discretizer.categories_[0]))

        for idx in leave_indices:
            discretized_features.append(
                X[:, idx].reshape(-1, 1).astype(float))
            discretized_feature_names.append(feature_names[idx])
            all_n_bins_to_use.append(1)

        return np.hstack(discretized_features), discretized_feature_names, \
            all_n_bins_to_use

    else:
        raise NotImplementedError


def argsort_feature_names(feature_names, heatmap, all_n_bins_to_use,
                          score_func, aggregate_func, reverse=False,
                          max_features_to_display=None,
                          cluster_mask=None):
    scores = []
    start_indices = []
    start_idx = 0
    for n_bins_to_use in all_n_bins_to_use:
        start_indices.append(start_idx)
        overall_score = None
        for row_idx in range(start_idx, start_idx + n_bins_to_use):
            if cluster_mask is None:
                score = score_func(heatmap[row_idx])
            else:
                score = score_func(heatmap[:, cluster_mask][row_idx])
            if overall_score is None:
                overall_score = score
            else:
                score = aggregate_func(overall_score, score)
        scores.append(overall_score)
        start_idx += n_bins_to_use
    scores = np.array(scores)

    if reverse:
        sort_indices = np.argsort(-scores)
    else:
        sort_indices = np.argsort(scores)

    new_order = []
    new_all_n_bins_to_use = []
    n_displayed_features = 0
    for old_feature_idx in sort_indices:
        n_bins_to_use = all_n_bins_to_use[old_feature_idx]

        n_displayed_features += n_bins_to_use
        if max_features_to_display is not None and \
                n_displayed_features > max_features_to_display:
            break

        start_idx = start_indices[old_feature_idx]
        for idx in range(n_bins_to_use):
            new_order.append(start_idx + idx)
        new_all_n_bins_to_use.append(n_bins_to_use)

    return new_order, new_all_n_bins_to_use


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
                 all_n_bins_to_use, max_features_to_display=None,
                 max_observed_times=None, save_filename=None, show_plot=True,
                 units=None, custom_xlabel=None, axhline_xmin=0,
                 cluster_sizes=None, cluster_size_threshold=None):
    feature_names = np.array(feature_names).copy()
    heatmap = heatmap.copy()

    # sort by max probability first (and then we do another sort)
    if cluster_size_threshold is not None:
        cluster_mask = (np.array(cluster_sizes) >= cluster_size_threshold)
    else:
        cluster_mask = None
    new_order, new_all_n_bins_to_use \
        = argsort_feature_names(feature_names, heatmap, all_n_bins_to_use,
                                np.max, max, reverse=True,
                                cluster_mask=cluster_mask)
    feature_names = feature_names[new_order]
    heatmap = heatmap[new_order]

    # sort by peak-to-peak
    new_order, new_all_n_bins_to_use \
        = argsort_feature_names(feature_names, heatmap, new_all_n_bins_to_use,
                                np.ptp, max, reverse=True,
                                max_features_to_display=max_features_to_display,
                                cluster_mask=cluster_mask)
    feature_names = feature_names[new_order]
    heatmap = heatmap[new_order]

    start_indices = []
    start_idx = 0
    for n_bins_to_use in new_all_n_bins_to_use:
        start_indices.append(start_idx)
        start_idx += n_bins_to_use

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

        ticks = list(cbar.get_ticks())
        heatmap_max = heatmap.max()
        for tick_idx, tick in enumerate(ticks):
            if tick >= heatmap_max:
                ticks = ticks[:tick_idx]
                break
        cbar.set_ticks(ticks + [heatmap.max()])

        for start_idx in start_indices[1:]:
            plt.axhline(y=start_idx, color='black', linestyle='dashed',
                        linewidth=.5, xmin=axhline_xmin, xmax=1, clip_on=False)

        plt.ylabel('Feature')
        if custom_xlabel is None:
            if units is None:
                xlabel = 'Clusters sorted by median survival time'
            else:
                xlabel = 'Clusters sorted by median survival time (%s)' % units
            if cluster_sizes is not None:
                xlabel += '; cluster sizes are stated in square brackets'
            plt.xlabel(xlabel)
        else:
            plt.xlabel(custom_xlabel)
        plt.yticks(np.arange(len(feature_names)) + 0.5,
                   list(feature_names),
                   rotation='horizontal')
        if max_observed_times is not None:
            if cluster_sizes is None:
                plt.xticks(np.arange(len(median_survival_times)) + 0.5,
                           [np.isinf(median_time) and (" >%.2f" % max_time)
                            or " %.2f " % median_time
                            for median_time, max_time
                            in zip(median_survival_times, max_observed_times)])
            else:
                plt.xticks(np.arange(len(median_survival_times)) + 0.5,
                           [(np.isinf(median_time) and (" >%.2f" % max_time)
                             or " %.2f " % median_time)
                            + "\n[%d]" % cluster_size
                            for median_time, max_time, cluster_size
                            in zip(median_survival_times,
                                   max_observed_times,
                                   cluster_sizes)])
        if save_filename is not None:
            plt.savefig(save_filename, bbox_inches='tight')
        if not show_plot:
            plt.close()
