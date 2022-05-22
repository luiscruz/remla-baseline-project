from dependencies.text_processors import bag_of_words, tfidf_features
import json

# Maybe change this test a little?


def test_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (bag_of_words(ex, words_to_index, 4) != ans).any():
            return False
    return True


def test_tfidf():

    with open("tests/dependencies/tfidf_process_data.json", "r") as file:
        validation_data = json.load(file)

    train_values = validation_data["train_values"]
    result_vocab = validation_data["result_vocab"]

    tfidf_vocabulary = tfidf_features(train_values["train"], train_values["val"], train_values["test"])[3]
    return tfidf_vocabulary == result_vocab
