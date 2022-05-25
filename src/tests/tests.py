"""Module to test ML pipeline logic"""
import unittest

from src.features.build_features import (
    tfidf_features,  # pylint: disable=no-name-in-module,import-error
)
from src.preprocess.preprocess_data import (  # pylint: disable=no-name-in-module,import-error
    init_data,
    text_prepare,
)


class TestPipeLine(unittest.TestCase):
    """Unit tests for the ML pipeline"""

    def test_text_prepare(self):

        """Test text preparation"""
        examples = [
            "SQL Server - any equivalent of Excel's CHOOSE function?",
            "How to free c++ memory vector<int> * arr?",
        ]
        answers = ["sql server equivalent excels choose function", "free c++ memory vectorint arr"]
        check = False
        wrong_case = None
        for ex, ans in zip(examples, answers):
            if text_prepare(ex) != ans:
                check = True
                wrong_case = ex

        message = f"Wrong answer for the case: '{wrong_case}'"
        self.assertFalse(check, message)

    def test_token(self):
        """Test tfidf"""
        root = "./data/raw/"
        X_train, X_val, X_test, _, _ = init_data(root + "train.tsv", root + "validation.tsv", root + "test.tsv")
        _, _, _, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
        tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}
        self.assertTrue("c#" in tfidf_vocab)
        # During the built-in tokenization of TfidfVectorizer and use ‘(\S+)’
        # regexp as a token_pattern in the constructor of the vectorizer.
        expected_tag = "c#"
        self.assertEqual(tfidf_reversed_vocab[4516], expected_tag)
