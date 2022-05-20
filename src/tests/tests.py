from src.features.build_features import *
from src.models.train_model import *
from src.models.predict_model import *

def test_text_prepare():
    """

    :return:
    """
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function",
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'

def test_my_bag_of_words(my_bad_of_words):
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (my_bag_of_words(ex, words_to_index, 4) != ans).any():
            return "Wrong answer for the case: '%s'" % ex
    
    print(test_my_bag_of_words())
    return 'Basic tests are passed.'

def test_token():
    X_train, X_val, X_test = split_data()
    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
    tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}
    return "c#" in tfidf_vocab

def test_token_after_transformation():
    """
    check whether you have c++ or c# in your vocabulary, as they are obviously important tokens in our tags prediction task:
    :return:
    """

    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
    tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}
    return tfidf_reversed_vocab[1879] == 'c#'

"""
y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
y_val_inversed = mlb.inverse_transform(y_val)
for i in range(3):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        X_val[i],
        ','.join(y_val_inversed[i]),
        ','.join(y_val_pred_inversed[i])
    ))
"""