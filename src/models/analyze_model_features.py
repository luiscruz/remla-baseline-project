"""Analyze model by finding top 5 positive and negative tags."""
import pickle
import sys

from src.config.definitions import ROOT_DIR


def print_words_for_tag(classifier_, tag_, tags_classes, index_to_words):
    """
        classifier: trained classifier
        tag: particular tag
        tags_classes: a list of classes names from MultiLabelBinarizer
        index_to_words: index_to_words transformation
        all_words: all words in the dictionary
        
        return nothing, just print top 5 positive and top 5 negative words for current tag
    """
    print('Tag:\t{}'.format(tag_))
    
    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator.
    
    model = classifier_.estimators_[tags_classes.index(tag_)]
    top_positive_words = [index_to_words[x] for x in model.coef_.argsort().tolist()[0][-5:]]
    top_negative_words = [index_to_words[x] for x in model.coef_.argsort().tolist()[0][:5]]
    
    print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
    print('Top negative words:\t{}\n'.format(', '.join(top_negative_words)))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        TAG = sys.argv[1]
    else:
        TAG = 'python' # default

    with open(ROOT_DIR / 'models/tfidf.pkl', 'rb') as f:
        classifier = pickle.load(f)
    with open(ROOT_DIR / 'models/mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)
    with open(ROOT_DIR / 'data/derivates/tfidf_vocab.pkl', 'rb') as f:
        tfidf_vocab = pickle.load(f)
    with open(ROOT_DIR / 'data/derivates/cleaned_train_dataset_properties.pkl', 'rb') as f:
        characteristics = pickle.load(f)

    tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}
    # ALL_WORDS = characteristics['ALL_WORDS']

    # print_words_for_tag(classifier, TAG, mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
	print_words_for_tag(classifier, TAG, mlb.classes, tfidf_reversed_vocab)

