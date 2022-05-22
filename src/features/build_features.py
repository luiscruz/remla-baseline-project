import numpy as np
import pandas as pd
# from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import sys

# For this project we will need to use a list of stop words. It can be downloaded from nltk:
# import nltk

# nltk.download('stopwords')
# from nltk.corpus import stopwords

# One of the most known difficulties when working with natural data is that it's unstructured.
# For example, if you use it "as is" and extract tokens just by splitting the titles by whitespaces,
# you will see that there are many "weird" tokens like *3.5?*, *"Flip*, etc.
# To prevent the problems, it's usually useful to prepare the data somehow.
# REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
# BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
# STOPWORDS = set(stopwords.words('english'))
# DICT_SIZE = 5000

"""
Task 1 - TextPrepare
"""


# def text_prepare(text):
#     """
#     text: a string

#     :return: modified initial string
#     """
#     text = text.lower()  # lowercase text
#     text = re.sub(REPLACE_BY_SPACE_RE, " ", text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
#     text = re.sub(BAD_SYMBOLS_RE, "", text)  # delete symbols which are in BAD_SYMBOLS_RE from text
#     text = " ".join([word for word in text.split() if not word in STOPWORDS])  # delete stopwords from text
#     return text


# def preprocess_text_prepare(X_train, X_val, X_test):
#     """

#     :param X_train:
#     :param X_val:
#     :param X_test:
#     :return:
#     """

#     X_train = [text_prepare(x) for x in X_train]
#     X_val = [text_prepare(x) for x in X_val]
#     X_test = [text_prepare(x) for x in X_test]
#     return X_train, X_val, X_test


# def text_prepare_tests():
#     prepared_questions = []
#     for line in open('data/text_prepare_tests.tsv', encoding='utf-8'):
#         line = text_prepare(line.strip())
#         prepared_questions.append(line)
#     text_prepare_results = '\n'.join(prepared_questions)
#     return text_prepare_results


"""
Task 2 - WordsTagsCount

We are assuming that tags_counts and words_counts are dictionaries like {'some_word_or_tag': frequency}. 
After applying the sorting procedure, results will be look like this: [('most_popular_word_or_tag', frequency), ('less_popular_word_or_tag', frequency), ...]. 
The grader gets the results in the following format (two comma-separated strings with line break):
"""


def word_tags_count(X_train, y_train):
    """
    :param: X_train
    :param: y_train
    :return: tags_counts, words_counts
    """
    # Dictionary of all tags from train corpus with their counts.
    tags_counts = {}
    # Dictionary of all words from train corpus with their counts.
    words_counts = {}

    for sentence in X_train:
        for word in sentence.split():
            if word in words_counts:
                words_counts[word] += 1
            else:
                words_counts[word] = 1

    for tags in y_train:
        for tag in tags:
            if tag in tags_counts:
                tags_counts[tag] += 1
            else:
                tags_counts[tag] = 1

    print(sorted(words_counts, key=words_counts.get, reverse=True)[:3])
    # We are assuming that tags_counts and words_counts are dictionaries like {'some_word_or_tag': frequency}. After
    # applying the sorting procedure, results will be look like this: [('most_popular_word_or_tag', frequency),
    # ('less_popular_word_or_tag', frequency), ...]

    return tags_counts, words_counts


"""
Bag of words

One of the well-known approaches is a bag-of-words representation. To create this transformation, follow the steps:
    Find N most popular words in train corpus and numerate them. Now we have a dictionary of the most popular words.
    For each title in the corpora create a zero vector with the dimension equals to N.
    For each text in the corpora iterate over words which are in the dictionary and increase by 1 the corresponding coordinate.

Let's try to do it for a toy example. Imagine that we have N = 4 and the list of the most popular words is ['hi', 
'you', 'me', 'are'] 

Then we need to numerate them, for example, like this: {'hi': 0, 'you': 1, 'me': 2, 'are': 3}

And we have the text, which we want to transform to the vector: 'hi how are you'

For this text we create a corresponding zero vector: [0, 0, 0, 0]

And iterate over all words, and if the word is in the dictionary, we increase the value of the corresponding position 
in the vector: 'hi':  [1, 0, 0, 0] 'how': [1, 0, 0, 0] # word 'how' is not in our dictionary 'are': [1, 0, 0, 
1] 'you': [1, 1, 0, 1] 

The resulting vector will be: [1, 1, 0, 1]
"""


# def my_bag_of_words(text, words_to_index, dict_size):
#     """
#     text: a string
#     dict_size: size of the dictionary
        
#     :return: a vector which is a bag-of-words representation of 'text'
#     """
#     result_vector = np.zeros(dict_size)

#     for word in text.split():
#         if word in words_to_index:
#             result_vector[words_to_index[word]] += 1
#     return result_vector


"""
TF-IDF

The second approach extends the bag-of-words framework by taking into account total frequencies of words in the corpora. 
It helps to penalize too frequent words and provide better features space.
Implement function tfidf_features using class TfidfVectorizer from scikit-learn. 
Use train corpus to train a vectorizer. 
Suggested: filter out too rare words (occur less than in 5 titles) and 
too frequent words (occur more than in 90% of the titles). 
Also, use bigrams along with unigrams in your vocabulary.
"""

PATH_TO_TRAIN = 'data/processed/train_preprocessed.tsv'
PATH_TO_VAL = 'data/processed/validation_preprocessed.tsv'
PATH_TO_TEST = 'data/processed/test_preprocessed.tsv'

OUT_PATH_TRAIN = 'data/interim/train_featurized.tsv'
OUT_PATH_VAL = 'data/interim/validation_featurized.tsv'
OUT_PATH_TEST = 'data/interim/test_featurized.tsv'

def tfidf_features(X_train, X_val, X_test):
    """
    X_train, X_val, X_test — samples
    :return: TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result

    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2),
                                       token_pattern='(\S+)')  ####### YOUR CODE HERE #######

    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)

    return X_train, X_val, X_test

def mlb_y_data(y_train, y_val, tags_counts):
    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)
    return y_train, y_val

def main():
    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython src/features/build_features.py train-file-path validation-file-path test-file-path\n")
        sys.exit(1)

    train = pd.read_csv(sys.argv[1])
    val = pd.read_csv(sys.argv[2])
    test = pd.read_csv(sys.argv[3])

    X_train, y_train = train[['X_train']], train[['y_train']]
    X_val, y_val = val[['X_val']], val[['y_val']]
    X_test = test[['X_test']]

    tags_counts, _ = word_tags_count(X_train=X_train, y_train=y_train)
    y_train, y_val = mlb_y_data(y_train, y_val, tags_counts)

    X_train, X_val, X_test = tfidf_features(X_train, X_val, X_test)

    train[['X_train']] = X_train
    train[['y_train']] = y_train
    val[['X_val']] = X_val
    val[['y_val']] = y_val
    test[['X_test']] = X_test

    train.to_csv(OUT_PATH_TRAIN, sep='\t', index=False)
    val.to_csv(OUT_PATH_VAL, sep='\t', index=False)
    test.to_csv(OUT_PATH_TEST, sep='\t', index=False)

if __name__ == '__main__':
    main()