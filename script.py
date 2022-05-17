"""
TODO Summary.

TODO Description
"""

import re
from ast import literal_eval
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, recall_score, roc_auc_score
import nltk

nltk.download('stopwords')
np.random.seed(42)


def read_data(filename):
    """
    TODO Summary.

    TODO Description
    """
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


def text_prepare(text):
    """
    TODO Summary.

    TODO Description
    """
    REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))

    # lowercase text
    text = text.lower()
    # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)
    # delete symbols which are in BAD_SYMBOLS_RE from text
    text = re.sub(BAD_SYMBOLS_RE, "", text)
    # delete stopwords from text
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text


def test_text_prepare():
    """
    TODO Summary.

    TODO Description
    """
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function",
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'


def my_bag_of_words(text, words_to_index, dict_size):
    """
    TODO Summary.

    TODO Description
    """
    result_vector = np.zeros(dict_size)

    for word in text.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1
    return result_vector


def test_my_bag_of_words():
    """
    TODO Summary.

    TODO Description
    """
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (my_bag_of_words(ex, words_to_index, 4) != ans).any():
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'


def tfidf_features(X_train, X_val, X_test):
    """
    TODO Summary.

    TODO Description
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result

    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern=r'(\S+)')  # YOUR CODE HERE # # nosec B106

    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)

    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_


def train_classifier(X_train, y_train, penalty='l1', C=1):
    """
    TODO Summary.

    TODO Description
    """
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    clf = LogisticRegression(penalty=penalty, C=C, dual=False, solver='liblinear')
    clf = OneVsRestClassifier(clf)
    clf.fit(X_train, y_train)

    return clf


def print_words_for_tag(classifier, tag, tags_classes, index_to_words, all_words):
    """
    TODO Summary.

    TODO Description
    """
    print('Tag:\t{}'.format(tag))

    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator.

    model = classifier.estimators_[tags_classes.index(tag)]
    top_positive_words = [index_to_words[x] for x in model.coef_.argsort().tolist()[0][-5:]]
    top_negative_words = [index_to_words[x] for x in model.coef_.argsort().tolist()[0][:5]]

    print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
    print('Top negative words:\t{}\n'.format(', '.join(top_negative_words)))


def print_evaluation_scores(y_val, predicted):
    """
    TODO Summary.

    TODO Description
    """
    print('Accuracy score: ', accuracy_score(y_val, predicted))
    print('F1 score: ', f1_score(y_val, predicted, average='weighted'))
    print('Average precision score: ', average_precision_score(y_val, predicted, average='macro'))


def main():
    """Is the main function."""
    train = read_data('data/train.tsv')
    validation = read_data('data/validation.tsv')
    test = pd.read_csv('data/test.tsv', sep='\t')

    print(train.head())

    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values

    print(test_text_prepare())

    prepared_questions = []
    for line in open('data/text_prepare_tests.tsv', encoding='utf-8'):
        line = text_prepare(line.strip())
        prepared_questions.append(line)
    text_prepare_results = '\n'.join(prepared_questions)

    X_train = [text_prepare(x) for x in X_train]
    X_val = [text_prepare(x) for x in X_val]
    X_test = [text_prepare(x) for x in X_test]

    print(X_train[:3])

    # Dictionary of all tags from train corpus with their counts.
    tags_counts = {}
    # Dictionary of all words from train corpus with their counts.
    words_counts = {}

    for sentence in X_train:
        for word in sentence.split():
            if word in words_counts:
                words_counts[word] += 1
            else:
                words_counts[word] = 1

    for tags in y_train:
        for tag in tags:
            if tag in tags_counts:
                tags_counts[tag] += 1
            else:
                tags_counts[tag] = 1

    # print(tags_counts)
    # print(words_counts)

    print(sorted(words_counts, key=words_counts.get, reverse=True)[:3])

    most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    DICT_SIZE = 5000
    INDEX_TO_WORDS = sorted(words_counts, key=words_counts.get, reverse=True)[:DICT_SIZE]  # YOUR CODE HERE #
    WORDS_TO_INDEX = {word: i for i, word in enumerate(INDEX_TO_WORDS)}
    ALL_WORDS = WORDS_TO_INDEX.keys()

    print(test_my_bag_of_words())

    X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
    X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
    X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])
    print('X_train shape ', X_train_mybag.shape)
    print('X_val shape ', X_val_mybag.shape)
    print('X_test shape ', X_test_mybag.shape)

    row = X_train_mybag[10].toarray()[0]
    non_zero_elements_count = (row > 0).sum()  # YOUR CODE HERE #

    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
    tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}

    print(tfidf_vocab["c#"])

    print(tfidf_reversed_vocab[1879])

    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)

    classifier_mybag = train_classifier(X_train_mybag, y_train)
    classifier_tfidf = train_classifier(X_train_tfidf, y_train)

    y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
    y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

    y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
    y_val_inversed = mlb.inverse_transform(y_val)
    for i in range(3):
        print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
            X_val[i],
            ','.join(y_val_inversed[i]),
            ','.join(y_val_pred_inversed[i])
        ))

    print('Bag-of-words')
    print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
    print('Tfidf')
    print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)

    print(roc_auc_score(y_val, y_val_predicted_scores_mybag, multi_class='ovo'))

    print(roc_auc_score(y_val, y_val_predicted_scores_tfidf, multi_class='ovo'))

    # coefficients = [0.1, 1, 10, 100]
    # penalties = ['l1', 'l2']

    # for coefficient in coefficients:
    #     for penalty in penalties:
    #         classifier_tfidf = train_classifier(X_train_tfidf, y_train, penalty=penalty, C=coefficient)
    #         y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    #         y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
    #         print("Coefficient: {}, Penalty: {}".format(coefficient, penalty))
    #         print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)

    classifier_tfidf = train_classifier(X_train_tfidf, y_train, penalty='l2', C=10)
    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

    test_predictions = classifier_tfidf.predict(X_test_tfidf)  # YOUR CODE HERE #
    test_pred_inversed = mlb.inverse_transform(test_predictions)

    test_predictions_for_submission = '\n'.join('%i\t%s' % (i, ','.join(row)) for i, row in enumerate(test_pred_inversed))

    print_words_for_tag(classifier_tfidf, 'c', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
    print_words_for_tag(classifier_tfidf, 'c++', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
    print_words_for_tag(classifier_tfidf, 'linux', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)


if __name__ == "__main__":
    main()
