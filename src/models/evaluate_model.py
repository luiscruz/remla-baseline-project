from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score


def predict_labels_and_scores(
    classifier_tfidf,
    X_val_tfidf
):
    """

    :param X_train:
    :param X_val:
    :param X_test:
    :param y_train:
    :return:
    """
    # y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
    # y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

    # A look at how classifier, which uses TF-IDF, works for a few examples:
    # y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
    # y_val_inversed = mlb.inverse_transform(y_val)
    # for i in range(3):
    #     print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
    #         X_val[i],
    #         ','.join(y_val_inversed[i]),
    #         ','.join(y_val_pred_inversed[i])
    #     ))

    # alternative_preprocessing_models(X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, mlb)

    return y_val_predicted_labels_tfidf, y_val_predicted_scores_tfidf


# def print_words_for_tag(classifier, tag, tags_classes, index_to_words):
#     """
#     Look at the features (words or n-grams) that are used with the largest weights in your logistic regression model.
#     classifier: trained classifier
#     tag: particular tag
#     tags_classes: a list of classes names from MultiLabelBinarizer
#     index_to_words: index_to_words transformation
#     all_words: all words in the dictionary

#     :return: nothing, just print top 5 positive and top 5 negative words for current tag
#     """
#     print('Tag:\t{}'.format(tag))

#     # Extract an estimator from the classifier for the given tag.
#     # Extract feature coefficients from the estimator.

#     model = classifier.estimators_[tags_classes.index(tag)]
#     top_positive_words = [index_to_words[x] for x in model.coef_.argsort().tolist()[0][-5:]]
#     top_negative_words = [index_to_words[x] for x in model.coef_.argsort().tolist()[0][:5]]

#     print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
#     print('Top negative words:\t{}\n'.format(', '.join(top_negative_words)))


"""
Print out scores
"""


def print_evaluation_scores(y_val, predicted):
    """

    :param y_val:
    :param predicted:
    :return: Nothing, prints the evaluation scores
    """
    print('Accuracy score: ', accuracy_score(y_val, predicted))
    print('F1 score: ', f1_score(y_val, predicted, average='weighted'))
    print('Average precision score: ', average_precision_score(y_val, predicted, average='macro'))
    print('Recall score: ', recall_score(y_val, predicted, average='macro'))


def print_eval_scores(
    y_val,
    # y_val_predicted_labels_mybag,
    y_val_predicted_labels_tfidf
):
    """

    :param y_val:
    :param y_val_predicted_labels_mybag:
    :param y_val_predicted_labels_tfidf:
    :return: Nothing, prints the evaluation scores for bag of words and tfidf.
    """
    # print('Bag-of-words')
    # print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
    print('Tfidf')
    print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)


def roc_auc_scores(
    y_val,
    # y_val_predicted_scores_mybag,
    y_val_predicted_scores_tfidf
):
    """

    :param y_val:
    :param y_val_predicted_scores_mybag:
    :param y_val_predicted_scores_tfidf:
    :return: Nothing, prints roc score for bag of words and tfidf.
    """

    # mybag_roc = roc_auc(y_val, y_val_predicted_scores_mybag, multi_class='ovo')
    # print('Bag-of-words: ', mybag_roc)
    tfidf_roc = roc_auc(y_val, y_val_predicted_scores_tfidf, multi_class='ovo')
    print('Tfidf: ', tfidf_roc)


"""
Task 4 - MultilabelClassification
"""


def alternative_preprocessing_models(
    classifier_tfidf,
    X_train_tfidf,
    X_val_tfidf,
    X_test_tfidf,
    y_train,
    mlb
):
    """
    Once the evaluation is set up, experiment with training your classifiers. We will use *F1-score weighted* as an evaluation metric.
    Our recommendation:
    - compare the quality of the bag-of-words and TF-IDF approaches and chose one of them.
    - for the chosen one, try *L1* and *L2*-regularization techniques in Logistic Regression with different coefficients (e.g. C equal to 0.1, 1, 10, 100).
    :return:
    """

    # coefficients = [0.1, 1, 10, 100]
    # penalties = ['l1', 'l2']

    # for coefficient in coefficients:
    #     for penalty in penalties:
    #         classifier_tfidf = train_classifier(X_train_tfidf, y_train, penalty=penalty, C=coefficient)
    #         y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    #         y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
    #         print("Coefficient: {}, Penalty: {}".format(coefficient, penalty))
    #         print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)

    test_predictions = classifier_tfidf.predict(X_test_tfidf)
    test_pred_inversed = mlb.inverse_transform(test_predictions)

    test_predictions_for_submission = '\n'.join(
        '%i\t%s' % (i, ','.join(row)) for i, row in enumerate(test_pred_inversed))
    print(test_predictions_for_submission)

def main():
    pass

if __name__ == '__main__':
    main()
