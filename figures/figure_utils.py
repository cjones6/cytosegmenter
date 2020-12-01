from collections import Counter
import numpy as np


def compute_min_dists(cps1, cps2):
    """
    Compute max_{1<=i<=cps1[-1]} {min_{1<=j<=cps2[-1]} |cps1[i]-cps2[j]|}.

    :param cps1: First array of change points.
    :param cps2: Second array of change points.
    :return: Distance from a change point in cps1 to the nearest change point in cps2.
    """
    pairwise_dists = np.abs(np.subtract.outer(cps1, cps2))
    min_dists = np.min(pairwise_dists, axis=1)
    return min_dists


def fill_histogram(histograms, indices, hist_idx, normalize=True):
    """
    Compute a histogram and fill in the designated entry in the array histograms.

    :param histograms: Array that will contains histograms.
    :param indices: Array with values from which a histogram should be computed.
    :param hist_idx: Index of row that will get filled in in histograms.
    :param normalize: Whether to normalize the histogram.
    :return: histograms: The array containing histograms with row "hist_idx" filled in by creating a histogram from the
                         entries in "indices".
    """
    counts = Counter(indices)
    for key in counts:
        histograms[hist_idx, key] += counts[key]
    rowsum = sum(histograms[hist_idx, :])
    if rowsum != 0 and normalize is True:
        histograms[hist_idx, :] /= rowsum

    return histograms


def get_pop_hists(pop, ids, unique_pop=None, normalize=True):
    """
    Create histograms of particle types for every time point.

    :param pop: A list with the particle type for each particle.
    :param ids: ID corresponding to each time point.
    :param unique_pop: List of particle types to consider.
    :param normalize: Whether the histograms should be normalized so they sum to 1.
    :return: A tuple consisting of:

        * histograms: An array with one histogram for each 3-minute window.
        * pop_dict: A dictionary where the keys are the names of each particle type and the values denote the column
                       each particle corresponds to in the histograms.
    """
    if unique_pop is None:
        unique_pop = list(set(pop).difference(['unknown']))
    n_unique_pop = len(unique_pop)
    pop_dict = {unique_pop[i]: i for i in range(n_unique_pop)}

    ntimes = np.max(ids)
    histograms = np.zeros((ntimes, n_unique_pop))

    start = 0
    ids = np.array(ids)

    print('Computing histograms...')
    for i in range(ntimes):
        if i % 100 == 0:
            print('{0:.2f}'.format(i/ntimes*100), '% done \r', end='')
        id = ids[start]
        for j in range(start, len(ids)):
            if ids[j] != id:
                end = j
                break
        pop_vals = []
        for idx in range(start, end):
            if pop[idx] in pop_dict:
                pop_vals.append(pop_dict[pop[idx]])
        histograms = fill_histogram(histograms, pop_vals, i, normalize=normalize)
        start = end
    print('100.00 % done \t\t')

    return histograms, pop_dict
