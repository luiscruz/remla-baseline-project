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










# todo add more
