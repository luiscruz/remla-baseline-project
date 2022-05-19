"""Train script which trains a classifier using logistic regression, measures the evaluation scores and writes them to the output directory."""

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
from joblib import load, dump


def train_classifier(X_train, y_train, penalty='l1', C=1):
    """
    Train a classifier using logistic regression with the provided parameters.

    :param X_train: train data.
    :param y_train: multi-class targets for the train data.
    :param penalty: penalty added to the logistic regression model.
    :param C: parameter representing the inverse of regularization strength
    :return: the trained classifier.
    """
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    clf = LogisticRegression(penalty=penalty, C=C, dual=False, solver='liblinear')
    clf = OneVsRestClassifier(clf)
    clf.fit(X_train, y_train)

    return clf


def write_evaluation_scores(f_name, y_val, pred_labels, pred_scores):
    """
    Write the evaluation results for the given inputs to the given filename in the output folder.

    :param f_name: filename.
    :param y_val: target labels for validation set.
    :param pred_labels: predicted labels.
    :param pred_scores: predicted scores.
    :return:
    """
    with open('output/' + f_name, 'w') as f:
        f.write('Accuracy score: {}\n'.format(accuracy_score(y_val, pred_labels)))
        f.write('F1 score: {}\n'.format(f1_score(y_val, pred_labels, average='weighted')))
        f.write('Average precision score: {}\n'.format(average_precision_score(y_val, pred_labels, average='macro')))
        f.write('ROC AUC score: {}\n'.format(roc_auc_score(y_val, pred_scores, multi_class='ovo')))


def main():
    """Is the main function."""
    X_train_tfidf, y_train = load('output/train_tfidf.joblib')
    X_val_tfidf, y_val = load('output/validation_tfidf.joblib')

    # Dictionary of all tags from train corpus with their counts.
    tags_counts = {}

    for tags in y_train:
        for tag in tags:
            if tag in tags_counts:
                tags_counts[tag] += 1
            else:
                tags_counts[tag] = 1

    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)

    coefficient = 10
    penalty = 'l2'
    classifier_tfidf = train_classifier(X_train_tfidf, y_train, penalty=penalty, C=coefficient)

    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

    # print('Tfidf')
    # print("Coefficient: {}, Penalty: {}".format(coefficient, penalty))
    write_evaluation_scores("stats.txt", y_val, y_val_predicted_labels_tfidf, y_val_predicted_scores_tfidf)

    dump(classifier_tfidf, 'output/model_tfidf.joblib')


if __name__ == "__main__":
    main()
