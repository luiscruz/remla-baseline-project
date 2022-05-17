"""TODO Summary."""

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
from joblib import load, dump


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


def write_evaluation_scores(f_name, y_val, pred_labels, pred_scores):
    """
    TODO Summary.

    TODO Description
    """
    with open('output/' + f_name, 'w') as f:
        f.write('Accuracy score: {}\n'.format(accuracy_score(y_val, pred_labels)))
        f.write('F1 score: {}\n'.format(f1_score(y_val, pred_labels, average='weighted')))
        f.write('Average precision score: {}\n'.format(average_precision_score(y_val, pred_labels, average='macro')))
        f.write('ROC AUC score: {}\n'.format(roc_auc_score(y_val, pred_scores, multi_class='ovo')))


def main():
    """Is the main function."""
    X_train_tfidf, y_train = load('output/train_tfidf.joblib')
    X_val_tfidf, y_val = load('output/train_tfidf.joblib')

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
