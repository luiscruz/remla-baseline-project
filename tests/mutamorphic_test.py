from typing import Tuple, List
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Input phrases

# Function to change it with
# synonyms
# hypernyms
# hyponyms

# options: random seed

# Obtain results for set of phrases

# Compare the results

_stopwords = stopwords.words('english')
_pos_tag_map = {
    'NN': [wn.NOUN],
    'JJ': [wn.ADJ, wn.ADJ_SAT],
    'RB': [wn.ADV],
    'VB': [wn.VERB]
}


def convert_pos_tag(nltk_pos_tag):
    root_tag = nltk_pos_tag[0:2]
    try:
        _pos_tag_map[root_tag]
        return _pos_tag_map[root_tag]
    except KeyError:
        return ''


class Word:

    def __init__(self, value: str, pos_tag: str):
        self.value = value
        self.pos_tag = convert_pos_tag(pos_tag)
        self.is_stopword: bool = value.lower() in _stopwords

    @staticmethod
    def from_tuple(tuple: Tuple[str, str]):
        '''
        Creates an instance of ``Word`` from a (token, part-of-speech tag), as generated by the
        functions ``nltk.pos_tag(tokens)``.

                Parameters:
                        tuple (Tuple[str, str]): A tuple with as the first argument the text token,
                                                 and as its second argument, the tokens POS tag.

                Returns:
                        an instance of ``Word``
        '''
        return Word(tuple[0], tuple[1])


def generate_variants(input_sentence: str,
                      num_variants: int = 5,
                      num_replacements: int = 1) -> List[str]:
    tokens = word_tokenize(input_sentence)
    tokens_with_pos_tags = nltk.pos_tag(tokens)
    words = [Word.from_tuple(t) for t in tokens_with_pos_tags]


    return [] # TODO



print(generate_variants("Uploading files via JSON Post request to a Web Service provided by Teambox"))