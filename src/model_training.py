"""
    Main logic of this stage:
    * loads the loblibs
    * trains two classifiers; one for bag-of-words and one for tf-idf
    * dumps the models in a joblib
"""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

output_directory = "output"

def train_classifier(X_train, y_train, penalty='l1', C=1):
    """
      X_train, y_train — training data

      return: trained classifier
    """

    # Create and fit LogisticRegression wrapped into OneVsRestClassifier.

    clf = LogisticRegression(penalty=penalty, C=C, dual=False, solver='liblinear')
    clf = OneVsRestClassifier(clf)
    clf.fit(X_train, y_train)

    return clf


def main():
    """
        Main logic of this stage:
        * loads the loblibs
        * trains two classifiers; one for bag-of-words and one for tf-idf
        * dumps the models in a joblib
    """
    tags_counts = joblib.load(output_directory + "/tags_counts.joblib")
    X_train_mybag, X_train_tfidf, _, _ = joblib.load(output_directory + "/vectorized_x.joblib")
    y_train, y_val = joblib.load(output_directory + "/y_preprocessed.joblib")

    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)

    # Train the classifiers for different data transformations: bag-of-words and tf-idf.
    classifier_mybag = train_classifier(X_train_mybag, y_train)
    classifier_tfidf = train_classifier(X_train_tfidf, y_train)

    joblib.dump((classifier_mybag, classifier_tfidf), output_directory + "/classifiers.joblib")
    joblib.dump(y_val, output_directory + "/fitted_y_val.joblib")
    joblib.dump(mlb, output_directory + "/mlb.joblib")


if __name__ == "__main__":
    main()
