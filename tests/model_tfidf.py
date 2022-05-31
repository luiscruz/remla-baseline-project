from dependencies.model_functions import *
from joblib import dump, load
import os
import sys
from sklearn.preprocessing import MultiLabelBinarizer


sys.path.append(os.getcwd())


def test_model_tfidf():

    from src import p3_train

    text_process_data = load(f'tests/dependencies/models_data.joblib')

    tfidf_data = text_process_data["tfidf"]

    y_val = text_process_data["bag"]["y_val"]
    y_train = text_process_data["bag"]["y_train"]
    tags_counts = text_process_data["bag"]["tags_counts"]

    # Train the classifier
    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)

    # Get the models

    tfidf_classifier_1 = p3_train.train_tfidf(tfidf_data, y_train, 2801)
    tfidf_classifier_2 = p3_train.train_tfidf(tfidf_data, y_train, 1998)

    # Retrieve the validation data
    X_val_tfdif = tfidf_data["X_val"]

    # Run the predictions
    labels_1 = tfidf_classifier_1.predict(X_val_tfdif)
    labels_2 = tfidf_classifier_2.predict(X_val_tfdif)

    return get_diff_stats(labels_1, labels_2, y_val)


check_diff(test_model_tfidf())
