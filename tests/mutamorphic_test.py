from typing import Tuple, List, Dict, Iterable, Callable
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import random

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
        return not self.is_stopword and not self.pos_tag == '' and not len(self.variants) == 0

    def _get_variations(self) -> Dict[str, int]:
        """
        TODO
        returns dict. key: synonym/hypernym, count: how many times was this suggested by nltk
        """
        if self.is_stopword or self.pos_tag == '':
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
                if self.value != synonym_without_underscore.lower():
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
                    if self.value != hypernym_without_underscore.lower():
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
                             random_seed: int) -> List[List[Tuple[Word, str]]]: #TODO use random seed
    """
    TODO comments
    """

    assert num_replacements <= len(non_trivial_words), "not enough nontrivial words to replace"

    choices = set()

    for _ in range(num_variants):
        while True:
            words = random.sample(non_trivial_words, num_replacements)
            mutations = {(word, random.choice(word.variants.keys())) for word in words}
            mutations = frozenset(mutations)
            if mutations not in choices:
                choices.add(mutations)
                break

    result = [list(mutations) for mutations in choices]

    return result


def _select_mutations_most_common_first(non_trivial_words: List[Word],
                                        num_replacements: int,
                                        num_variants: int,
                                        random_seed: int) -> List[List[Tuple[Word, str]]]:
    """
    TODO comments
    """

    # Some preparations: group mutations by the number of times it is suggested by WordNet
    grouped_by_counts = dict()
    for word in non_trivial_words:
        for variant, count in word.variants.items():
            if count not in grouped_by_counts.keys():
                grouped_by_counts[count] = {(word, variant)}
            else:
                grouped_by_counts[count].add((word, variant))

    groups = list(grouped_by_counts.keys())
    groups.sort(reverse=True)

    # Choose the mutations. The mutations with the highest counts are chosen first. But for a single
    # word, only one mutation is chosen at a time. So if for a sentence, a mutation is chosen for
    # word X, if more mutations still need to be chosen, the mutation with the highest count for
    # word Y is chosen, where X =/= Y.
    mutations_list = []
    for _ in range(num_variants):
        mutations = []
        chosen_words = set()
        current_group_number = 0
        while len(chosen_words) < num_replacements:
            group = grouped_by_counts[groups[current_group_number]]
            candidates = [x for x in group if x[0] not in chosen_words]
            if len(candidates) > 0:
                mutation = random.choice(candidates)
                chosen_words.add(mutation[0])
                mutations.append(mutation)
                group.remove(mutation)
            else:
                current_group_number += 1
        mutations_list.append(mutations)

    return mutations_list


MUTATION_SELECTION_STRATEGIES: Dict[str, Callable[[List[Word], int, int, int], List[List[Tuple[Word, str]]]]] = {
    "random": _select_mutations_random,
    "most_common_first": _select_mutations_most_common_first,
}


def _get_selection_strategy_func(selection_strategy: str)\
        -> Callable[[List[Word], int, int, int], List[List[Tuple[Word, str]]]]:
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
    non_trivial_words = {word: index for (index, word) in enumerate(words) if word.is_nontrivial}

    selection_strategy_func = _get_selection_strategy_func(selection_strategy)

    mutations_list = selection_strategy_func(list(non_trivial_words.keys()),
                                             num_replacements,
                                             num_variants,
                                             random_seed)

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


test_sentence = "Uploading files via JSON Post request to a Web Service provided by Teambox"
result = mutate_by_replacement(test_sentence,
                               num_replacements=2,
                               num_variants=10,
                               selection_strategy="most_common_first")
print(f"ORIGINAL: \n{test_sentence}")
print("VARIANTS:")
for x in result:
    print(x)
