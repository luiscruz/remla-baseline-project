from typing import Tuple, List, Dict, Iterable, Callable
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

STOPWORDS = stopwords.words('english')
POST_TAG_MAP = {
    'NN': [wn.NOUN],
    'JJ': [wn.ADJ, wn.ADJ_SAT],
    'RB': [wn.ADV],
    'VB': [wn.VERB]
}


class Word:

    def __init__(self, value: str, pos_tag: str):
        self.value = value.lower()
        self.pos_tag = Word.convert_pos_tag(pos_tag)
        self.is_stopword: bool = value.lower() in STOPWORDS
        self.variants = self._get_variations()

    @property
    def is_nontrivial(self):
        return not self.is_stopword and not self.pos_tag == ''

    def _get_variations(self) -> Dict[str, int]:
        """
        TODO
        returns dict. key: synonym/hypernym, count: how many times was this suggested by nltk
        """
        if not self.is_nontrivial:
            return dict()

        synsets = wn.synsets(self.value, pos=self.pos_tag)

        synonyms = self._get_synonyms(synsets)
        hypernyms = self._get_hypernyms(synsets)

        result = Word._count_variants(synonyms, hypernyms)

        return result

    def _get_synonyms(self, synsets: List[Synset]) -> List[str]:
        """
        TODO
        returns list of synonym suggestions
        """
        result: List[str] = []
        for synset in synsets:
            for lemma in synset.lemmas():
                substrings = lemma.name().split('.')
                synonym = substrings[-1]
                synonym_without_underscore = re.sub(r'_', ' ', synonym)
                if self.value != synonym_without_underscore:
                    result.append(synonym_without_underscore)

        return result

    def _get_hypernyms(self, synsets: List[Synset]) -> List[str]:
        """
        TODO
        returns list of hypernym suggestions
        """
        result: List[str] = []
        for synset in synsets:
            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemmas():
                    substrings = lemma.name().split('.')
                    hypernym = substrings[-1]
                    hypernym_without_underscore = re.sub(r'_', ' ', hypernym)
                    if self.value != hypernym_without_underscore:
                        result.append(hypernym_without_underscore)

        return result

    @staticmethod
    def convert_pos_tag(nltk_pos_tag):
        root_tag = nltk_pos_tag[0:2]
        if root_tag in POST_TAG_MAP.keys():
            return POST_TAG_MAP[root_tag]
        else:
            return ''

    @staticmethod
    def _count_variants(*args: Iterable[str]) -> Dict[str, int]:
        """
        TODO
        counts occurences of variation suggestions
        """
        result: Dict[str, int] = dict()

        for variants in args:
            for variant in variants:
                if variant not in result.keys():
                    result[variant] = 1
                else:
                    result[variant] += 1

        return result

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

    def __repr__(self):
        return f"Word(\"{self.value}\", tag: {self.pos_tag}, stopword: {self.is_stopword})"


def _select_mutations_random(non_trivial_words: List[Word],
                             num_replacements: int,
                             num_variants: int,
                             random_seed: int) -> List[List[Tuple[Word, str]]]:
    """
    TODO comments
    """

    pass


def _select_mutations_most_common_first(non_trivial_words: List[Word],
                                        num_replacements: int,
                                        num_variants: int,
                                        random_seed: int) -> List[List[Tuple[Word, str]]]:
    """
    TODO comments
    """

    pass


MUTATION_SELECTION_STRATEGIES: Dict[str, Callable[[List[Word], int, int, int], List[List[Tuple[Word, str]]]]] = {
    "random": _select_mutations_random,
    "most_common_first": _select_mutations_most_common_first,
}


def _get_selection_strategy_func(selection_strategy: str)\
        -> Callable[[List[Word], int, int], List[List[Tuple[Word, str]]]]:
    """
    TODO comments
    """

    assert selection_strategy in MUTATION_SELECTION_STRATEGIES\
        , "Unknown mutation selection strategy."

    return MUTATION_SELECTION_STRATEGIES[selection_strategy]


def mutate_by_replacement(input_sentence: str,
                          num_replacements: int = 1,
                          num_variants: int = 5,
                          selection_strategy: str = "random",
                          random_seed: int = 13) -> List[str]:
    """
    TODO comments
    """

    tokens = word_tokenize(input_sentence)
    tokens_with_pos_tags = nltk.pos_tag(tokens)
    words = [Word.from_tuple(t) for t in tokens_with_pos_tags]
    non_trivial_words = {word: index for (word, index) in enumerate(words) if word.is_nontrivial}

    selection_strategy_func = _get_selection_strategy_func(selection_strategy)

    mutations_list = selection_strategy_func(list(non_trivial_words.keys()),
                                             num_replacements,
                                             num_variants)

    mutated_sentences = []
    for mutations in mutations_list:
        sentence = [word.value for word in words]
        for mutation in mutations:
            word = mutation[0]
            replacement = mutation[1]
            index = non_trivial_words[word]
            sentence[index] = replacement
        mutated_sentences.append(" ".join(sentence))

    return mutated_sentences


print(mutate_by_replacement("Uploading files via JSON Post request to a Web Service provided by Teambox"))
