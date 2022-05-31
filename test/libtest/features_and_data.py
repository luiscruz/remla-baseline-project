"""
    ML Test library functions for features and data.
    Based on section 2 of the paper referenced below.

    Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, D. Sculley (2016). What’s your ML test score? A rubric for ML production systems. Reliable Machine Learning in the Wild - NIPS 2016 Workshop (2016).
    Available: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45742.pdf
"""
import collections

import numpy as np


def no_unsuitable_features(used_features, unsuitable_features):
    """
        Compares the list of used features to the list of unsuitable features. The size of the intersection should be 0.
    """
    illegal_features = [f for f in used_features if f in unsuitable_features]
    assert len(illegal_features) == 0, "At least one unsuitable feature is used."


def feature_target_correlations(dataset, target):
    """"
        Takes a matrix (#datapoints, #features) and a vector of targets (#datapoints).
        Calculates the correlation of each individual feature with the target.
    """
    n, f = dataset.shape
    correlations = []

    # Loop over each feature
    for i in range(f):
        all_occurrences_of_feature_i = dataset[:, i]
        corr = np.corrcoef(all_occurrences_of_feature_i.toarray().reshape(-1), target)[0][1]
        correlations.append(corr)

    # None of the correlations should be exactly 0
    assert all(correlations), "At least one feature has 0 correlation with the target."


def pairwise_feature_correlations(dataset):
    """"
        Takes a matrix (#datapoints, #features).
        Calculates the correlation of each pair of features.
    """
    n, f = dataset.shape
    correlations = []

    # Loop over each pair of features
    for i in range(f):
        for j in range(i):
            all_occurrences_of_feature_i = dataset[:, i]
            all_occurrences_of_feature_j = dataset[:, j]
            corr = np.corrcoef(all_occurrences_of_feature_i.toarray().reshape(-1),
                               all_occurrences_of_feature_j.toarray().reshape(-1))[0][1]
            correlations.append(corr)

    # None of the correlations are exactly 1
    assert all([c != 1.0000 for c in correlations]), "At least one pair of features has perfect correlation."


def feature_values(dataset, feature_column_id, expected_values):
    arr = dataset[:, feature_column_id].toarray().reshape(-1)
    for i in set(arr):
        assert i in expected_values


def top_feature_values(dataset, feature_column_id, expected_values, topK=2, at_least_top_k_account_for=0.5):
    arr = dataset[:, feature_column_id].toarray().reshape(-1)
    n_data = len(arr)
    features_distribution = collections.Counter(arr).most_common()
    top_features = features_distribution[:topK]
    summation = 0
    for l, r in top_features:
        print(l, r / n_data)
        assert l in expected_values
        summation += r
    assert summation / n_data >= at_least_top_k_account_for

# todo add more
