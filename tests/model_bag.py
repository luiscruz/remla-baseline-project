from joblib import dump, load
import os
from sklearn.preprocessing import MultiLabelBinarizer
from dependencies.model_functions import *

import sys


sys.path.append(os.getcwd())

# This test needs to run the preprocessing first!!


def test_model_bag():

    from src import p3_train

    text_process_data = load(f'tests/dependencies/models_data.joblib')

    bag_of_words_data = text_process_data["bag"]

    y_val = bag_of_words_data["y_val"]
    y_train = bag_of_words_data["y_train"]
    tags_counts = bag_of_words_data["tags_counts"]

    # Train the classifier
    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)

    # Get the models
    bag_classifier_1 = p3_train.train_bag(bag_of_words_data, y_train, 2801)
    bag_classifier_2 = p3_train.train_bag(bag_of_words_data, y_train, 1998)

    # Retrieve the validation data
    X_val_bag = text_process_data["bag"]["X_val"]

    # Run the predictions
    labels_bag_1 = bag_classifier_1.predict(X_val_bag)
    labels_bag_2 = bag_classifier_2.predict(X_val_bag)

    return get_diff_stats(labels_bag_1, labels_bag_2, y_val)


check_diff(test_model_bag())
