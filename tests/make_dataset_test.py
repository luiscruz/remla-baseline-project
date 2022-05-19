from src.data import make_dataset


def test_text_prepare_1():
    example = "SQL Server - any equivalent of Excel's CHOOSE function?"
    answer = "sql server equivalent excels choose function"

    assert make_dataset.text_prepare(example) == answer


def test_text_prepare_2():
    example = "How to free c++ memory vector<int> * arr?"
    answer = "free c++ memory vectorint arr"
    assert make_dataset.text_prepare(example) == answer
