from src.model import bag_of_words

def test_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    
    for ex, ans in zip(examples, answers):
        if (bag_of_words.bag_of_words(ex, words_to_index, 4) != ans).any():
            return f"Wrong answer for the case: '{ex}'"

    return 'Basic tests are passed.'
