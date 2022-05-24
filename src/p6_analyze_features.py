from joblib import dump, load
import os
import json

RESULTS_DIRECTION_BAG = "results/popular_words_bag.json"
RESULTS_DIRECTION_TFIDF = "results/popular_words_tfidf.json"


def get_list_words(tuple_list):
    words = list()
    for elem in tuple_list:
        words.append(elem[0])
    return words


def dump_feat_results_bag(classifier, tag, tags_classes, index_to_words):

    model = classifier.estimators_[tags_classes.index(tag)]

    most_seen = [index_to_words[x][0] for x in model.coef_.argsort().tolist()[0][:5]]
    less_seen = [index_to_words[x][0] for x in model.coef_.argsort().tolist()[0][-5:]]

    if os.path.exists(RESULTS_DIRECTION_BAG):
        creation_time = int(os.path.getctime(RESULTS_DIRECTION_BAG)*10e5)
        os.rename(RESULTS_DIRECTION_BAG, RESULTS_DIRECTION_BAG.replace(".json", f"-{creation_time}.json"))

    with open(f"{RESULTS_DIRECTION_BAG}", "w") as file_write:
        file_write.write(json.dumps({"most_seen": most_seen, "least_seen": less_seen}, indent=4))


def dump_feat_results_tfidf(classifier, tag, tags_classes, tfidf_vocab):

    model = classifier.estimators_[tags_classes.index(tag)]

    top_negative_words = [tfidf_vocab[x] for x in model.coef_.argsort().tolist()[0][:5]]
    top_positive_words = [tfidf_vocab[x] for x in model.coef_.argsort().tolist()[0][-5:]]

    if os.path.exists(RESULTS_DIRECTION_TFIDF):
        creation_time = int(os.path.getctime(RESULTS_DIRECTION_TFIDF)*10e5)
        os.rename(RESULTS_DIRECTION_TFIDF, RESULTS_DIRECTION_TFIDF.replace(
            ".json", f"{creation_time}.json"))
    with open(f"{RESULTS_DIRECTION_TFIDF}", "w") as file_write:
        file_write.write(json.dumps({"top_positive_words": top_positive_words,
                         "top_negative_words": top_negative_words}, indent=4))


mlb = load("output/multi_label_binarizer.joblib")
words_dictionaries = load("output/words_dictionaries.joblib")
classifiers = load("output/classifiers.joblib")
tfidf_vocab = load("output/text_processor_data.joblib")["tfidf"]["tfidf_vocab"]
tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}

classifier_bag = classifiers["bag"]
dump_feat_results_bag(classifier_bag, 'c', mlb.classes,
                      words_dictionaries["vocabulary"])

classifier_tfidf = classifiers["tfidf"]
dump_feat_results_tfidf(classifier_tfidf, 'c', mlb.classes,
                        tfidf_reversed_vocab)
