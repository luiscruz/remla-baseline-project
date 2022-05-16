from joblib import dump, load
import os
import json

RESULTS_DIRECTION = "results/popular_words.json"


def get_list_words(tuple_list):
    words = list()
    for elem in tuple_list:
        words.append(elem[0])
    return words


def dump_feat_results(results):

    if os.path.exists(RESULTS_DIRECTION):
        creation_time = int(os.path.getctime(RESULTS_DIRECTION)*10e5)
        os.rename(RESULTS_DIRECTION, RESULTS_DIRECTION.replace("popular_words.", f"popular_words-{creation_time}.json"))
    with open(f"{RESULTS_DIRECTION}", "w") as file_write:
        file_write.write(json.dumps(results, indent=4))


def print_words_for_tag(classifier, tag, tags_classes, index_to_words, all_words):
    """
        classifier: trained classifier
        tag: particular tag
        tags_classes: a list of classes names from MultiLabelBinarizer
        index_to_words: index_to_words transformation
        all_words: all words in the dictionary

        return nothing, just print top 5 positive and top 5 negative words for current tag
    """
    print('Tag:\t{}'.format(tag))

    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator.

    model = classifier.estimators_[tags_classes.index(tag)]

    most_seen = [index_to_words[x][0] for x in model.coef_.argsort().tolist()[0][:5]]
    less_seen = [index_to_words[x][0] for x in model.coef_.argsort().tolist()[0][-5:]]

    print('Top 5 most seen words words:\t{}'.format(', '.join(most_seen)))
    print('Top 5 least seen word words:\t{}\n'.format(', '.join(less_seen)))

    dump_feat_results({"most_seen": most_seen, "least_seen": less_seen})


mlb = load("output/multi_label_binarizer.joblib")
words_dictionaries = load("output/words_dictionaries.joblib")
classifier = load("output/classifier.joblib")

print_words_for_tag(classifier, 'c', mlb.classes, words_dictionaries["vocabulary"], words_dictionaries["all_words"])
