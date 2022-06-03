"""
    ML Testing the StackOverflow label predictor for features and data. Making use of the [todo library name] library.
"""
import joblib
import numpy as np

import libtest.features_and_data as lib
from src.text_preprocessing import text_prepare
from src.vectorization import my_bag_of_words


def test_no_unsuitable_features():
    lib.no_unsuitable_features(['title'], [])


def prepare_correlation_analysis():
    mybag, tfidf, _, _ = joblib.load("output/vectorized_x.joblib")
    y_train, _ = joblib.load("output/y_preprocessed.joblib")

    unique_labels = {None}
    for i in y_train:
        for j in i:
            unique_labels.add(j)

    labels_id, id_labels = {}, {}
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

    return mybag, tfidf, labels_matrix


# def test_pairwise_feature_correlations():
#     mybag, tfidf, _ = prepare_correlation_analysis()
#
#     lib.pairwise_feature_correlations(mybag, sample_size=100000)
#     lib.pairwise_feature_correlations(tfidf, sample_size=100000)


def test_feature_target_correlations():
    mybag, tfidf, labels_matrix = prepare_correlation_analysis()

    for i in range(3):
        lib.feature_target_correlations(mybag, labels_matrix[:, i])
        lib.feature_target_correlations(tfidf, labels_matrix[:, i])


def test_feature_values():
    mybag, tfidf, _, _ = joblib.load("output/vectorized_x.joblib")
    import collections
    features_values = {}
    features_distribution = {}

    for i in range(mybag.shape[1]):
        all_occurrences_of_feature_i = mybag[:, i]
        arr = all_occurrences_of_feature_i.toarray().reshape(-1)
        features_values[i] = set(arr)
        features_distribution[i] = collections.Counter(arr).most_common()

    expected = [0, 1, 2, 3, 4, 5, 6]
    for i in features_values.keys():
        for k in features_values[i]:
            assert k in expected
        break


def test_preprocessing_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function",
               "free c++ memory vectorint arr"]

    lib.preprocessing_validation(examples, answers, text_prepare)


def test_preprocessing_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]

    lib.preprocessing_validation(examples, answers,
                                 lambda x: my_bag_of_words(x, words_to_index, 4),
                                 equals=lambda a, b: (a == b).all())


# todo add more
