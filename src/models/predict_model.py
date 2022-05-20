from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from src.models.train_model import *
from src.features.build_features import *

def print_evaluation_scores(y_val, predicted):
    """

    :param y_val:
    :param predicted:
    :return: Nothing, prints the evaluation scores
    """
    print('Accuracy score: ', accuracy_score(y_val, predicted))
    print('F1 score: ', f1_score(y_val, predicted, average='weighted'))
    print('Average precision score: ', average_precision_score(y_val, predicted, average='macro'))


def print_eval_scores(y_val, y_val_predicted_labels_mybag, y_val_predicted_labels_tfidf):
    """

    :param y_val:
    :param y_val_predicted_labels_mybag:
    :param y_val_predicted_labels_tfidf:
    :return: Nothing, prints the evaluation scores for bag of words and tfidf.
    """
    print('Bag-of-words')
    print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
    print('Tfidf')
    print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)


def roc_auc_scores(y_val, y_val_predicted_scores_mybag, y_val_predicted_scores_tfidf):
    """

    :param y_val:
    :param y_val_predicted_scores_mybag:
    :param y_val_predicted_scores_tfidf:
    :return: Nothing, prints roc score for bag of words and tfidf.
    """

    mybag_roc = roc_auc(y_val, y_val_predicted_scores_mybag, multi_class='ovo')
    print('Bag-of-words: ', mybag_roc)
    tfidf_roc = roc_auc(y_val, y_val_predicted_scores_tfidf, multi_class='ovo')
    print('Tfidf: ', tfidf_roc)


def predict_labels_and_scores(X_train, X_val, X_test, y_train):
    """

    :param X_train:
    :param X_val:
    :param X_test:
    :param y_train:
    :return:
    """
    X_train_mybag, X_val_mybag, X_test_mybag = train_mybag(X_train, X_val, X_test, y_train)
    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)

    y_train, y_val, clf = train_classifier(X_train, y_train)

    classifier_mybag, classifier_tfidf = train_classifier_for_transformations(X_train_mybag, y_train)
    y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
    y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

    return y_val_predicted_labels_mybag, y_val_predicted_scores_mybag, y_val_predicted_labels_tfidf, y_val_predicted_scores_tfidf


def test_predications(X_train_tfidf, X_val_tfidf, y_train):
    """

    :param X_train_tfidf:
    :param X_val_tfidf:
    :param y_train:
    :return:
    """
    classifier_tfidf = train_classifier(X_train_tfidf, y_train, penalty='l2', C=10)
    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)


def print_words_for_tag(classifier, tag, tags_classes, index_to_words, all_words):
    """
    Look at the features (words or n-grams) that are used with the largest weights in your logistic regression model.
    classifier: trained classifier
    tag: particular tag
    tags_classes: a list of classes names from MultiLabelBinarizer
    index_to_words: index_to_words transformation
    all_words: all words in the dictionary

    :return: nothing, just print top 5 positive and top 5 negative words for current tag
    """
    print('Tag:\t{}'.format(tag))

    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator.

    model = classifier_tfidf.estimators_[tags_classes.index(tag)]
    top_positive_words = [index_to_words[x] for x in model.coef_.argsort().tolist()[0][-5:]]
    top_negative_words = [index_to_words[x] for x in model.coef_.argsort().tolist()[0][:5]]

    print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
    print('Top negative words:\t{}\n'.format(', '.join(top_negative_words)))
