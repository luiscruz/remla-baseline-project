from src.features import build_features


def test_count_words_strings():
    test_failed = False
    test_message = ""
    test_string = ["da da dum da da dum di"]
    test_res = {"da": 4, "dum": 2, "di": 1}
    func_res = build_features.count_words_strings(test_string)

    for k in func_res.keys():
        if test_res[k] != func_res[k]:
            test_failed = True
            test_message += (
                "test failed for key ["
                + str(k)
                + "]. Counted ["
                + str(func_res[k])
                + "], but should be ["
                + str(test_res[k])
                + "]"
            )
            break

    assert not test_failed, test_message


def test_count_words_lists():
    test_failed = False
    test_message = ""

    test_list = [["da", "da", "dum", "da", "da", "dum", "di"]]
    test_res = {"da": 4, "dum": 2, "di": 1}
    func_res = build_features.count_words_lists(test_list)

    for k in func_res.keys():
        if test_res[k] != func_res[k]:
            test_failed = True
            test_message += (
                "test failed for key ["
                + str(k)
                + "]. Counted ["
                + str(func_res[k])
                + "], but should be ["
                + str(test_res[k])
                + "]"
            )
            break

    assert not test_failed, test_message


def test_my_bag_of_words():
    test_failed = False

    words_to_index = {"hi": 0, "you": 1, "me": 2, "are": 3}
    examples = ["hi how are you"]
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (build_features.my_bag_of_words(ex, words_to_index, 4) != ans).any():
            test_failed = True
            break

    assert not test_failed
