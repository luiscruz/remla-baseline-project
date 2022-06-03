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
    assert len(illegal_features) == 0


def feature_target_correlations(dataset, target, sample_size=10000):
    """"
        Takes a matrix (#datapoints, #features) and a vector of targets (#datapoints).
        Calculates the correlation of each individual feature with the target.
        A sample of the points is taken for speedup
    """
    n, f = dataset.shape

    # Assure that sample_size is not too big -> out of bounds
    if n < sample_size:
        sample_size = n

    correlations = np.corrcoef(np.transpose(dataset[:sample_size].toarray()), target[:sample_size])

    print("correlation coefficients calculated")

    # None of the correlations should be exactly 0
    assert np.all(correlations), "At least one feature has 0 correlation with the target. " \
                                 "Perhaps increasing the sample_size solves the issue."


def pairwise_feature_correlations(dataset, sample_size=10000, feature_sample=5):
    """"
        Takes a matrix (#datapoints, #features).
        Calculates the correlation of each pair of features.
    """
    n, f = dataset.shape

    # Assure that sample_size is not too big -> out of bounds
    if n < sample_size:
        sample_size = n
    if f < feature_sample:
        feature_sample = f

    correlations = np.corrcoef(np.transpose(dataset[:sample_size, :feature_sample].toarray()),
                               np.transpose(dataset[:sample_size, :feature_sample].toarray()))

    # Matrix is 4 concatenations of the matrix of interest, chop off
    correlations = correlations[:feature_sample, :feature_sample]
    # Delete 1.0 from the diagonal (because correlations with itself is always 1.0, we are not interested in that)
    correlations -= np.eye(correlations.shape[0])

    assert 1.0000000 not in correlations, "At least one pair of features has perfect correlation. " \
                                          "Perhaps increasing the sample_size solves the issue." \
                                          "Feature pairs with perfect correlation:" \
                                          f"{list(zip(np.where(correlations == 1.0)[0], np.where(correlations == 1.0)[1]))}"


def preprocessing_validation(examples, answers, preprocess_function, equals=lambda a, b: a == b):
    """
        Asserts that preprocessing works.
    """
    for ex, ans in zip(examples, answers):
        assert equals(preprocess_function(ex), ans), f"Preprocessing went wrong for {ex}"


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
