from src.text_preprocessing import my_bag_of_words, text_prepare


def test_my_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        assert not (my_bag_of_words(ex, words_to_index, 4) != ans).any(), "Wrong answer for the case: '%s'" % ex


def test_text_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function",
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        assert not (text_prepare(ex) != ans), "Wrong answer for the case: '%s'" % ex
