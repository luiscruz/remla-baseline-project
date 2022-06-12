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
    Test that a model does not contain any features that have been manually determined as unsuitable for use.
    Compares the list of used features to the list of unsuitable features. The size of the intersection should be 0.
    :param used_features: list of used features
    :param unsuitable_features: list of features manually determened to be unsuitable
    :return: list of illegal features
    """
    illegal_features = [f for f in used_features if f in unsuitable_features]
    assert len(illegal_features) == 0


def feature_target_correlations(dataset, target, sample_size=10000):
    """
    Test the relationship between each feature and the target.
    Calculates the correlation of each individual feature with the target.
    :param dataset:  a matrix (#datapoints, #features)
    :param target:  a vector of targets (#datapoints)
    :param sample_size: size of samples
    :return: the correlation between each feature and target
    """
    n, f = dataset.shape

    # Assure that sample_size is not too big -> out of bounds
    if n < sample_size:
        sample_size = n

    correlations = np.corrcoef(np.transpose(dataset[:sample_size].toarray()), target[:sample_size])

    # None of the correlations should be exactly 0
    assert np.all(correlations), "At least one feature has 0 correlation with the target. " \
                                 "Perhaps increasing the sample_size solves the issue."


def pairwise_feature_correlations(dataset, sample_size=10000, feature_sample=5):
    """
    Test the pairwise correlations between individual features.
    Calculates the correlation of each pair of features.
    :param dataset: matrix (#datapoints, #features)
    :param sample_size: size of samples
    :param feature_sample: number of feature samples
    :return: the correlation between each pair of features
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
        Test all code that creates input features, both in training and serving.
        Asserts that preprocessing works.
    :param examples: example data input
    :param answers: expected data output after preprocessing
    :param preprocess_function: preprocessing function
    :param equals: equals function to be used for assertion, default is lambda
    :return: whether examples are expected answers
    """
    for ex, ans in zip(examples, answers):
        assert equals(preprocess_function(ex), ans), f"Preprocessing went wrong for {ex}"


def feature_values(dataset, feature_column_id, expected_values):
    """
    Test that the distributions of each feature match your expectations.
    :param dataset:
    :param feature_column_id:
    :param expected_values: expected output
    :return: whether data in feature column are as expected output
    """
    arr = dataset[:, feature_column_id].toarray().reshape(-1)
    for i in set(arr):
        assert i in expected_values


def top_feature_values(dataset, feature_column_id, expected_values, topK=2, at_least_top_k_account_for=0.5):
    """
    Test the expected top feature values with actual data.
    :param dataset:
    :param feature_column_id:
    :param expected_values: expected output
    :param topK: number of top feature values to be tested
    :param at_least_top_k_account_for: percentage the data should be as expected output
    :return: whether percentage of top features in actual data is higher than at_least_top_k_account_for
    """
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

