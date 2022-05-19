from models.train_model import train_classifier, train_classifier_for_transformations
from src.features.build_features import tfidf_features, train_mybag

def predict_labels_and_scores(X_train, X_val, X_test):
    X_train_mybag, X_val_mybag, X_test_mybag = train_mybag()
    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)

    y_train, y_val, clf = train_classifier(X_train, y_train)

    classifier_mybag, classifier_tfidf= train_classifier_for_transformations(X_train_mybag, y_train)
    y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
    y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
    
    return y_val_predicted_labels_mybag, y_val_predicted_scores_mybag, y_val_predicted_labels_tfidf, y_val_predicted_scores_tfidf

def test_predications():
    classifier_tfidf = train_classifier(X_train_tfidf, y_train, penalty='l2', C=10)
    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)