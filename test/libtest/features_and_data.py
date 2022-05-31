"""
    ML Test library functions for features and data.
    Based on section 2 of the paper referenced below.

    Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, D. Sculley (2016). What’s your ML test score? A rubric for ML production systems. Reliable Machine Learning in the Wild - NIPS 2016 Workshop (2016).
    Available: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45742.pdf
"""
from scipy.stats.stats import pearsonr


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
        corr = pearsonr(all_occurrences_of_feature_i, target)[0]
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
            corr = pearsonr(all_occurrences_of_feature_i, all_occurrences_of_feature_j)[0]
            correlations.append(corr)

    # None of the correlations are exactly 1
    assert all([c != 1.0000 for c in correlations]), "At least one pair of features has perfect correlation."

# todo add more
