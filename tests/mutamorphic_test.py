from joblib import load
import json
import random
import sys
import os
from dependencies.model_functions import *
import numpy as np

sys.path.append(os.getcwd()+"/mutamorfic")
sys.path.append(os.getcwd())

SEED = 600
random.seed(SEED)
LIMIT = 2000


def load_test_phrases(limit=LIMIT):

    from src import p1_preprocessing

    validation_data = p1_preprocessing.read_data("data/validation.tsv")

    X_val, y_val = validation_data['title'].values, validation_data['tags'].values

    lower_limit = random.randint(0, len(X_val)-limit-1)
    upper_limit = lower_limit+limit

    return X_val[lower_limit:upper_limit], y_val[lower_limit:upper_limit]


def load_x_train():
    from src import p1_preprocessing

    return p1_preprocessing.read_data(f'data/train.tsv')['title'].values


def change_phrases(X_val, num_replacements=2, num_variants=3, selection_strategy="random"):
    from mutamorfic.mutators import ReplacementMutator

    mutator = ReplacementMutator(num_replacements, num_variants, selection_strategy)

    new_x_vals = []
    for i in range(0, num_variants):
        new_x_vals.append(list())

    for sentence in X_val:
        results = mutator.mutate(sentence, SEED)

        for i in range(0, num_variants):

            if len(results) == num_variants:
                new_x_vals[i].append(results[i])
            else:
                # This way we ENSURE that both results have equal length
                new_x_vals[i].append(sentence)

    return new_x_vals
    # To be implemented


def get_values_classifiers(X_val, y_val, X_train):
    from src import p1_preprocessing
    from src import p2_text_processors
    X_val = [p1_preprocessing.text_prepare(x) for x in X_val]

    DICT_SIZE, INDEX_TO_WORDS = p2_text_processors.get_tags_and_words(X_val, y_val)[:2]

    tfidf_vectorizer = p2_text_processors.TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2),
                                                          token_pattern='(\S+)')

    X_train = [p1_preprocessing.text_prepare(x) for x in X_train]

    X_val_bag = p2_text_processors.get_x_values_bag(INDEX_TO_WORDS, DICT_SIZE, X_val)
    tfidf_vectorizer.fit_transform(X_train)

    X_val_tfidf = tfidf_vectorizer.transform(X_val)

    return X_val_bag, X_val_tfidf


def test_mutamorphic():

    from src import p2_text_processors

    X_val, y_val = load_test_phrases()
    X_train = load_x_train()

    classifiers = load("tests/dependencies/classifiers.joblib")
    y_val = load('tests/dependencies/val_data.joblib')[:LIMIT]
    bag_classifier = classifiers["bag"]
    tfidf_classifier = classifiers["tfidf"]

    # First we ge the unaltered labels

    X_val_bag, X_val_tfidf = get_values_classifiers(X_val, y_val, X_train)

    labels_bag_og = bag_classifier.predict(X_val_bag)
    labels_tfidf_og = tfidf_classifier.predict(X_val_tfidf)

    # Now we go for the mutants
    X_val_mutates = change_phrases(X_val)

    for X_val_mutate in X_val_mutates:
        X_val_bag, X_val_tfidf = get_values_classifiers(np.array(X_val_mutate), y_val, X_train)

        labels_bag = bag_classifier.predict(X_val_bag)
        labels_tfidf = tfidf_classifier.predict(X_val_tfidf)

        check_diff(get_diff_stats(labels_bag_og, labels_bag, y_val))
        check_diff(get_diff_stats(labels_tfidf_og, labels_tfidf, y_val))

    print("All tests passed.")


test_mutamorphic()
