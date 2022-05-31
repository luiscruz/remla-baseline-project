from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from joblib import dump, load

# Trainning function

OUTPUT_DIR = "output/"


def train_classifier(X_train, y_train, penalty='l1', C=1, seed=10):
    """
      X_train, y_train — training data

      return: trained classifier
    """

    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    clf = LogisticRegression(penalty=penalty, C=C, dual=False, solver='liblinear', random_state=seed)
    clf = OneVsRestClassifier(clf)
    clf.fit(X_train, y_train)

    return clf


def train_bag(bag_of_words_data, y_train, seed=10):
    X_train = bag_of_words_data["X_train"]
    classifier = train_classifier(X_train, y_train, seed=seed)
    return classifier


def train_tfidf(tfidif_data, y_train, seed=10):

    X_train = tfidif_data["X_train"]
    classifier = train_classifier(X_train, y_train, seed=seed)
    return classifier


def main():
    text_process_data = load('output/text_processor_data.joblib')

    bag_of_words_data = text_process_data["bag"]
    tfidif_data = text_process_data["tfidf"]

    y_val = bag_of_words_data["y_val"]
    y_train = bag_of_words_data["y_train"]
    tags_counts = bag_of_words_data["tags_counts"]

    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)

    bag_classifier = train_bag(bag_of_words_data, y_train)
    tfidf_classifier = train_tfidf(tfidif_data, y_train)

    classifiers = {"bag": bag_classifier, "tfidf": tfidf_classifier}

    dump(classifiers, f'{OUTPUT_DIR}classifiers.joblib')
    dump(mlb, f"{OUTPUT_DIR}multi_label_binarizer.joblib")
    dump(y_val, f'{OUTPUT_DIR}val_data.joblib')


if __name__ == "__main__":
    main()
