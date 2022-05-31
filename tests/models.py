from joblib import dump, load
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
import sys


sys.path.append(os.getcwd())

# This test needs to run the preprocessing first!!

LIMIT = 0.1


def test_accuracy_models():

    from src import p3_train
    from src import p1_preprocessing
    from src import p2_text_processors

    preprocessed_data = p1_preprocessing.get_preprocessed_data()
    text_process_data = p2_text_processors.get_processors(preprocessed_data)[0]

    bag_of_words_data = text_process_data["bag"]
    tfidif_data = text_process_data["tfidf"]

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

    tfidf_classifier_1 = p3_train.train_tfidf(tfidif_data, y_train, 2801)
    tfidf_classifier_2 = p3_train.train_tfidf(tfidif_data, y_train, 1998)

    # Retrieve the validation data
    X_val_bag = text_process_data["bag"]["X_val"]
    X_val_tfdif = text_process_data["tfidf"]["X_val"]

    # Run the predictions
    labels_bag_1 = bag_classifier_1.predict(X_val_bag)
    labels_bag_2 = bag_classifier_2.predict(X_val_bag)

    diff_acc_bag = abs(accuracy_score(y_val, labels_bag_1)-accuracy_score(y_val, labels_bag_2))

    labels_tfidf_1 = tfidf_classifier_1.predict(X_val_tfdif)
    labels_tfidf_2 = tfidf_classifier_2.predict(X_val_tfdif)

    diff_acc_tfidf = abs(accuracy_score(y_val, labels_tfidf_1)-accuracy_score(y_val, labels_tfidf_2))

    return diff_acc_bag < LIMIT or diff_acc_tfidf < LIMIT


assert test_accuracy_models(), "The models are not working properly"
