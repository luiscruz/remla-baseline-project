"""
    ML Test library functions for model development.
    Based on section 3 of the paper referenced below.

    Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, D. Sculley (2016). What’s your ML test score? A rubric for ML production systems. Reliable Machine Learning in the Wild - NIPS 2016 Workshop (2016).
    Available: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45742.pdf
"""


def compare_against_baseline(model = "linear", own_score,  train_X, train_Y, test_X, test_Y):
    options = ["linear", "logistic"]
    if model not in options:
        model = "linear"
        
    # TRAINS Classifier

    # TESTS Classifier & RETURNS score

