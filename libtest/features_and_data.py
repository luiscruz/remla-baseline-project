"""
    ML Test library functions for features and data.
    Based on section 2 of the paper referenced below.

    Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, D. Sculley (2016). What’s your ML test score? A rubric for ML production systems. Reliable Machine Learning in the Wild - NIPS 2016 Workshop (2016).
    Available: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45742.pdf
"""


def no_unsuitable_features(used_features, unsuitable_features):
    """
        Compares the list of used features to the list of unsuitable features. The size of the intersection should be 0.
    """
    illegal_features = [f for f in used_features if f in unsuitable_features]
    assert len(illegal_features) == 0

# todo add more
