from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from joblib import dump, load

# Trainning function


def train_classifier(X_train, y_train, penalty='l1', C=1):
    """
      X_train, y_train — training data

      return: trained classifier
    """

    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    clf = LogisticRegression(penalty=penalty, C=C, dual=False, solver='liblinear')
    clf = OneVsRestClassifier(clf)
    clf.fit(X_train, y_train)

    return clf


def main():
    bag_of_words_data = load('output/bag_of_words.joblib')

    X_train = bag_of_words_data["X_train"]
    tags_counts = bag_of_words_data["tags_counts"]
    y_val = bag_of_words_data["y_val"]
    y_train = bag_of_words_data["y_train"]

    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)

    classifier = train_classifier(X_train, y_train)

    dump(mlb, "output/multi_label_binarizer.joblib")
    dump(classifier, 'output/classifier.joblib')
    dump(y_val, 'output/val_data.joblib')


if __name__ == "__main__":
    main()
