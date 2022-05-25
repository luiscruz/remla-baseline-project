import unittest
from src.features.build_features import text_prepare


class TestBuildFeatures(unittest.TestCase):

    def test_text_prepare(self):
        examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                    "How to free c++ memory vector<int> * arr?"]
        answers = ["sql server equivalent excels choose function",
                   "free c++ memory vectorint arr"]
        for ex, ans in zip(examples, answers):
            if text_prepare(ex) != ans:
                return "Wrong answer for the case: '%s'" % ex
        return 'Basic tests are passed.'


if __name__ == '__main__':
    unittest.main()
