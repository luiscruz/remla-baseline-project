from typing import Dict


def get_corpus_counts(X_train: list[str], y_train: list[str]):
    tags_counts: Dict[str, int] = {}
    words_counts: Dict[str, int] = {}

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

    return words_counts, tags_counts
