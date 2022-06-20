"""
Survival analysis metrics

Author: George H. Chen (georgechen [at symbol] cmu.edu)
"""
import numpy as np
from lifelines.utils import concordance_index
from pycox.utils import idx_at_times
from pycox.evaluation.concordance import concordance_td


def neg_cindex(y_true, risk_scores):
    return -concordance_index(y_true[:, 0], -risk_scores, y_true[:, 1])


def neg_cindex_td(y_true, surv_pred, exact=True, n_subsamples=None,
                  approx_batch_size=2**14):
    surv, times = surv_pred
    if exact:
        return -concordance_td(y_true[:, 0], y_true[:, 1], surv,
                               idx_at_times(times, y_true[:, 0], 'post'),
                               'antolini')

    # note that for small datasets, no subsampling will actually be done by the
    # code below, so the calculation will still be exact
    n_samples = y_true.shape[0]
    if n_subsamples is not None:
        batch_size = int(np.ceil(n_samples / n_subsamples))
    else:
        if n_samples < approx_batch_size:
            batch_size = n_samples
        else:
            n_subsamples = int(np.ceil(n_samples / approx_batch_size))
            batch_size = int(np.ceil(n_samples / n_subsamples))

    c_indices = []
    for batch_start_idx in range(0, n_samples, batch_size):
        batch_end_idx = min(batch_start_idx + batch_size, n_samples)
        c_indices.append(
            concordance_td(y_true[batch_start_idx:batch_end_idx, 0],
                           y_true[batch_start_idx:batch_end_idx, 1],
                           surv[:, batch_start_idx:batch_end_idx],
                           idx_at_times(
                               times,
                               y_true[batch_start_idx:batch_end_idx, 0],
                               'post'),
                           'antolini'))
    return -np.mean(c_indices)
