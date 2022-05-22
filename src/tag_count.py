def get_tag_count(X_train, y_train):

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
    # most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    # most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    return words_counts, tags_counts