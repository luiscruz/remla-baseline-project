import sys
import os
import json

sys.path.append(os.getcwd())


def test_tfidf():

    from src import p2_text_processors

    with open("tests/dependencies/tfidf_process_data.json", "r") as file:
        validation_data = json.load(file)

    train_values = validation_data["train_values"]
    result_vocab = validation_data["result_vocab"]

    tfidf_vocabulary = p2_text_processors.tfidf_features(
        train_values["train"], train_values["val"], train_values["test"])[3]

    return tfidf_vocabulary == result_vocab


assert test_tfidf(), "TFIDF is not working properly"
