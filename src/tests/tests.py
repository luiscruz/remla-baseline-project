from src.features.build_features import *
from src.models.train_model import *
from src.models.predict_model import *

import unittest

class TestPipeLine(unittest.TestCase):
        
    # Text prepare test
    def test_text_prepare(self):
        examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                    "How to free c++ memory vector<int> * arr?"]
        answers = ["sql server equivalent excels choose function",
                "free c++ memory vectorint arr"]
        check = False
        wrong_case = None
        for ex, ans in zip(examples, answers):
            if text_prepare(ex) != ans:
                check = True
                wrong_case = ex

        message = "Wrong answer for the case: '%s'" % wrong_case 
        self.assertTrue(check, message)

    # Bag-of-words content test
    def test_my_bag_of_words(self):
        words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
        examples = ['hi how are you']
        answers = [[1, 1, 0, 1]]
        check = False
        wrong_case = None
        for ex, ans in zip(examples, answers):
            if (my_bag_of_words(ex, words_to_index, 4) != ans).any():
                check = True
                wrong_case = ex
                
        message = "Wrong answer for the case: '%s'" % wrong_case
        self.assertTrue(check, message)


    # Bag-of-words amount test
    def test_amount_bag_of_words(self):
        X_train_mybag = []
        # For the 11th row in X_train_mybag find how many non-zero elements it has. In this task the answer (variable
        # non_zero_elements_count) should be a number, e.g. 20.
        row = X_train_mybag[10].toarray()[0]
        non_zero_elements_count = (row > 0).sum()  ####### YOUR CODE HERE #######
        self.assertEquals(non_zero_elements_count, 20)


    # TFIDF test contains #c
    def test_token(self):
        X_train, X_val, X_test, y_train, y_val = init_data()
        X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
        tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}
        self.assertTrue("c#" in tfidf_vocab)

    # TFIDF test 1879 is #c
    def test_token_after_transformation(self):
        """
        check whether you have c++ or c# in your vocabulary, as they are obviously important tokens in our tags prediction task:
        :return:
        """
        X_train, X_val, X_test, y_train, y_val = init_data()
        X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
        tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}
        self.assertEquals(tfidf_reversed_vocab[1879], 'c#')
