import nltk
from ast import literal_eval
import pandas as pd
import numpy as np
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data

def process_question(text):
    """
        text: a string
        
        return: modified initial string
    """
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    text = text.lower() # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, "", text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join([word for word in text.split() if not word in STOPWORDS]) # delete stopwords from text
    return text

def count_tags_and_words(X_train, y_train):
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
    return (tags_counts, words_counts)

def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
    
    
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1,2), token_pattern='(\S+)') ####### YOUR CODE HERE #######
    
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)
    
    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_

def preprocess_data(input_dir, output_dir):
    train = read_data(input_dir + '/train.tsv')
    validation = read_data(input_dir + '/validation.tsv')
    test = pd.read_csv(input_dir + '/test.tsv', sep='\t')

    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values

    X_train = [process_question(x) for x in X_train]
    X_val = [process_question(x)  for x in X_val]
    X_test = [process_question(x)  for x in X_test]

    X_train, X_val, X_test, vocab = tfidf_features(X_train, X_val, X_test)
    reversed_vocab = {i:word for word,i in vocab.items()}

    (tags_counts, word_counts) = count_tags_and_words(X_train, y_train)

    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)

    X_train.to_csv(output_dir + '/X_train.tsv', sep='\t')
    X_test.to_csv(output_dir + '/X_test.tsv', sep='\t')
    y_train.to_csv(output_dir + '/y_train.tsv', sep='\t')
    y_val.to_csv(output_dir + '/y_val.tsv', sep='\t')

if __name__ == "__main__":
    # execute only if run as the entry point into the program
    preprocess_data(input_dir="../../data/raw", output_dir="../../data/data/processed")