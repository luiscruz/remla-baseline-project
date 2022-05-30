from dependencies.bag_of_words import bag_of_words

def test_bag():

    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (bag_of_words(ex, words_to_index, 4) != ans).any():
            return False
    return True


print(test_bag())
