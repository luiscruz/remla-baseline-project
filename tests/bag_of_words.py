import sys
import os
from joblib import load, dump
import json
import numpy as np

sys.path.append(os.getcwd())


# Change this test completly


def test_bag_of_words():

    from src import p2_text_processors
    from src import p1_preprocessing

    bag_data = load("tests/dependencies/bag_data.joblib")

    words_to_index = bag_data["words_to_index"]
    training_data = bag_data["training_data"]
    answers = bag_data["answers"]

    results = []

    for phrase in training_data:

        result = p2_text_processors.bag_of_words(phrase, words_to_index, len(words_to_index))
        results.append(np.count_nonzero(result))

    return results == answers


assert test_bag_of_words(), "Bag of words is not working properly"
