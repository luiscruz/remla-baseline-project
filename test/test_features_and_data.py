"""
    ML Testing the StackOverflow label predictor for features and data. Making use of the [todo library name] library.
"""
import joblib
import numpy as np

import libtest.features_and_data
from src.text_preprocessing import feature_list


def test_no_unsuitable_features():
    libtest.features_and_data.no_unsuitable_features(feature_list, [])


def test_hey():
    mybag, tfidf, _, _ = joblib.load("../output/vectorized_x.joblib")
    y_train, _ = joblib.load("../output/y_preprocessed.joblib")
    # print("mybag",mybag.shape)
    # print("tfidf",tfidf.shape)
    # print("ytrain",y_train)
    unique_labels = {None}
    for i in y_train:
        for j in i:
            unique_labels.add(j)

    print(unique_labels)
    print(len(unique_labels))
    labels_id = {}
    id_labels = {}
    counter = 0
    for i in unique_labels:
        labels_id[counter] = i
        id_labels[i] = counter
        counter += 1

    labels_matrix = np.zeros([mybag.shape[0], len(unique_labels)])
    print(labels_matrix.shape)

    for i in range(len(y_train)):
        for j in y_train[i]:
            labels_matrix[i][id_labels[j]] = 1

    libtest.features_and_data.feature_target_correlations(mybag, labels_matrix[:, 1])
    libtest.features_and_data.pairwise_feature_correlations(mybag)
    # libtest.features_and_data.feature_target_correlations(tfidf, labels_matrix[:, 1])
    # libtest.features_and_data.pairwise_feature_correlations(tfidf)


def test_hey2():
    mybag, tfidf, _, _ = joblib.load("../output/vectorized_x.joblib")
    import collections
    features_values = {}
    features_distribution = {}

    for i in range(mybag.shape[1]):
        all_occurrences_of_feature_i = mybag[:, i]
        arr = all_occurrences_of_feature_i.toarray().reshape(-1)
        features_values[i] = set(arr)
        features_distribution[i] = collections.Counter(arr).most_common()

    print(features_values)
    print(features_distribution)
    expected = [1,2,3,4,5,6]
    for i in features_values.keys():
        for k in features_values[i]:
            assert k in expected
        break


# todo add more
