from src.preprocess import text_prepare


def test_text_prepare():
    examples = [
        "SQL Server - any equivalent of Excel's CHOOSE function?",
        "How to free c++ memory vector<int> * arr?",
    ]
    answers = [
        "sql server equivalent excels choose function",
        "free c++ memory vectorint arr",
    ]
    for ex, ans in zip(examples, answers):
        assert text_prepare(ex) == ans
