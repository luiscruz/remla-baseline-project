from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from text_preprocessing import get_train_test_data


def train_classifier(X_train, y_train, penalty='l1', C=1):
    """
      X_train, y_train - training data
      return: trained classifier
    """

    solver = 'liblinear'
    random_seed = 42
    penalty = 'l1'

    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
    clf = LogisticRegression(penalty=penalty, C=C, dual=False, solver=solver, random_state=random_seed)
    clf = OneVsRestClassifier(clf)
    clf.fit(X_train, y_train)

    return clf

def get_classifiers(data_dir):
    (X_train, y_train, X_val, y_val, X_test, X_train_mybag, X_val_mybag, X_test_mybag, X_train_tfidf, X_val_tfidf,
     X_test_tfidf, tfidf_vectorizer, tags_counts, words_to_index, dict_size) = get_train_test_data(data_dir)

    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)
    print("Training classifier for train dataset")
    classifier_mybag = train_classifier(X_train_mybag, y_train)
    print("Training classifier for test dataset")
    classifier_tfidf = train_classifier(X_train_tfidf, y_train)

    return classifier_mybag, classifier_tfidf, y_train, y_val, mlb, tfidf_vectorizer, words_to_index, dict_size
