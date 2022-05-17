from common_tools import delete_import, copy_file_import


if copy_file_import("src/2_bag_of_words.py"):
    from testing_file import bag_of_words

delete_import()


def test_bag():

    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (bag_of_words(ex, words_to_index, 4) != ans).any():
            return False
    return True


print(test_bag())
