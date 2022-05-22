import re
import numpy as np
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# For this project we will need to use a list of stop words. It can be downloaded from nltk:
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# One of the most known difficulties when working with natural data is that it's unstructured.
# For example, if you use it "as is" and extract tokens just by splitting the titles by whitespaces,
# you will see that there are many "weird" tokens like *3.5?*, *"Flip*, etc.
# To prevent the problems, it's usually useful to prepare the data somehow.
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
DICT_SIZE = 5000

"""
Task 1 - TextPrepare
"""


def text_prepare(text):
    """
    text: a string

    :return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, "", text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join([word for word in text.split() if not word in STOPWORDS])  # delete stopwords from text
    return text


def preprocess_text_prepare(X_train, X_val, X_test):
    """

    :param X_train:
    :param X_val:
    :param X_test:
    :return:
    """

    X_train = [text_prepare(x) for x in X_train]
    X_val = [text_prepare(x) for x in X_val]
    X_test = [text_prepare(x) for x in X_test]
    return X_train, X_val, X_test


def text_prepare_tests():
    prepared_questions = []
    for line in open('data/text_prepare_tests.tsv', encoding='utf-8'):
        line = text_prepare(line.strip())
        prepared_questions.append(line)
    text_prepare_results = '\n'.join(prepared_questions)
    return text_prepare_results


"""
Task 2 - WordsTagsCount

We are assuming that tags_counts and words_counts are dictionaries like {'some_word_or_tag': frequency}. 
After applying the sorting procedure, results will be look like this: [('most_popular_word_or_tag', frequency), ('less_popular_word_or_tag', frequency), ...]. 
The grader gets the results in the following format (two comma-separated strings with line break):
"""


def word_tags_count(X_train, y_train):
    """

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

    # print(tags_counts)
    # print(words_counts)

    print(sorted(words_counts, key=words_counts.get, reverse=True)[:3])
    # We are assuming that tags_counts and words_counts are dictionaries like {'some_word_or_tag': frequency}. After
    # applying the sorting procedure, results will be look like this: [('most_popular_word_or_tag', frequency),
    # ('less_popular_word_or_tag', frequency), ...]

    return tags_counts, words_counts


def most_common_counts(tags_counts, words_counts):
    """
    :param tags_counts:
    :param words_counts:
    :return: Most common counts of the tags and word counts.
    """
    most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    return most_common_tags, most_common_words


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


def my_bag_of_words(text, words_to_index, dict_size):
    """
    text: a string
    dict_size: size of the dictionary
        
    :return: a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)

    for word in text.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1
    return result_vector


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

    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_
