"""
    ML Testing the StackOverflow label predictor for features and data. Making use of the [todo library name] library.
"""
import libtest.features_and_data
from src.text_preprocessing import feature_list


def test_no_unsuitable_features():
    libtest.features_and_data.no_unsuitable_features(feature_list, [])

# todo add more
