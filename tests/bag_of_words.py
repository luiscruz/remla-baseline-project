import sys
import os


sys.path.append(os.getcwd())


# Maybe change this test a little?


def test_bag_of_words():

    from src import p2_text_processors

    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (p2_text_processors.bag_of_words(ex, words_to_index, 4) != ans).any():
            return False
    return True


assert test_bag_of_words(), "Bag of words is not working properly"
