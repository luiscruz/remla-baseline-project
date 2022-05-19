from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.metrics import roc_auc_score as roc_auc

from model import get_classifiers
from text_preprocessing import get_train_test_data


def print_evaluation_scores(y_val, predicted):
    print('Accuracy score: ', accuracy_score(y_val, predicted))
    print('F1 score: ', f1_score(y_val, predicted, average='weighted'))
    print('Average precision score: ', average_precision_score(y_val, predicted, average='macro'))


def bag_of_words_tfidf_evaluation():
    classifier_mybag, classifier_tfidf, y_train, y_val = get_classifiers()
    X_train, _, X_val, _, X_test, X_train_mybag, X_val_mybag, X_test_mybag, X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab, _, words_to_index, dict_size = get_train_test_data(data=3)
    y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
    y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
    print('Bag-of-words')
    print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
    print("roc_acu: ", roc_auc(y_val, y_val_predicted_scores_mybag, multi_class='ovo'))
    print('Tfidf')
    print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)
    print("roc_acu: ", roc_auc(y_val, y_val_predicted_scores_tfidf, multi_class='ovo'))
