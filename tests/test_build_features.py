import pytest
from src.features.build_features import text_prepare


class TestBuildFeatures:

    def test_text_prepare(self):
        examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                    "How to free c++ memory vector<int> * arr?"]
        answers = ["sql server equivalent excels choose function",
                   "free c++ memory vectorint arr"]
        for ex, ans in zip(examples, answers):
            assert text_prepare(ex) == ans


if __name__ == '__main__':
    pytest.main()
