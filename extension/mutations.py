from typing import Tuple, List, Dict, Callable
from extension.Word import Word
from nltk.tokenize import word_tokenize
import nltk
import random as random_pkg
from random import Random


def _select_mutations_random(non_trivial_words: List[Word],
                             num_replacements: int,
                             num_variants: int,
                             rng: Random) -> List[List[Tuple[Word, str]]]:
    """
    Selects mutations at random. Ensures that each output sentence is unique, although mutations
    at the nontrivial word level might be reused across output sentences. For each output sentence,
    words are selected randomly, without replacement, to be mutated. Then, for the selected words,
    a variant (synonym/hypernym) is chosen at random (uniform dist.) from the available variants of
    that word.

            Parameters:
                ``nontrivial_words`` (``List[Word]``): a list of all nontrivial words (words that
                can be replaced by a variant (synonym/hypernym)

                ``num_replacements`` (``int``): the number of words that should be replaced per
                output sentence

                ``num_variants`` (``int``): the number of output sentences

                ``rng`` (``Random``): the random generator instance used to make random choices

            Returns:
                A list of mutations to be made for each output sentence. For each output sentence,
                the function returns a list of tuples representing the mutation. The first element
                of the tuple is the nontrivial word that is to be replaced. The second element of
                the tuple is the variant that was chosen for the word as a replacement.
    """
    assert num_replacements <= len(non_trivial_words), "not enough nontrivial words to replace"

    choices = set()

    for _ in range(num_variants):
        while True:
            words = rng.sample(non_trivial_words, num_replacements)
            mutations = {(word, rng.choice(word.variants.keys())) for word in words}
            mutations = frozenset(mutations)
            if mutations not in choices:
                choices.add(mutations)
                break

    mutations_list = [list(mutations) for mutations in choices]

    return mutations_list


def _select_mutations_most_common_first(nontrivial_words: List[Word],
                                        num_replacements: int,
                                        num_variants: int,
                                        rng: Random) -> List[List[Tuple[Word, str]]]:
    """
    Selects mutations based on how many times a variant for a nontrivial word was suggested by
    WordNet. The mutations with the highest counts are chosen first. But for a single word, only one
    mutation is chosen at a time. So if for a sentence, a mutation is chosen for word X, when more
    mutations still need to be chosen, the mutation with the highest count for word Y is chosen,
    where X =/= Y. If variants with equal counts are considered, a uniformly random choice is made.

            Parameters:
                ``nontrivial_words`` (``List[Word]``): a list of all nontrivial words (words that
                can be replaced by a variant (synonym/hypernym)

                ``num_replacements`` (``int``): the number of words that should be replaced per
                output sentence

                ``num_variants`` (``int``): the number of output sentences

                ``rng`` (``Random``): the random generator instance used to make random choices

            Returns:
                A list of mutations to be made for each output sentence. For each output sentence,
                the function returns a list of tuples representing the mutation. The first element
                of the tuple is the nontrivial word that is to be replaced. The second element of
                the tuple is the variant that was chosen for the word as a replacement.
    """

    # Some preparations: group mutations by the number of times it is suggested by WordNet
    grouped_by_counts = dict()
    for word in nontrivial_words:
        for variant, count in word.variants.items():
            if count not in grouped_by_counts.keys():
                grouped_by_counts[count] = {(word, variant)}
            else:
                grouped_by_counts[count].add((word, variant))

    groups = list(grouped_by_counts.keys())
    groups.sort(reverse=True)

    # Choose the mutations.
    mutations_list = []
    for _ in range(num_variants):
        mutations = []
        chosen_words = set()
        current_group_number = 0
        while len(chosen_words) < num_replacements:
            group = grouped_by_counts[groups[current_group_number]]
            candidates = [x for x in group if x[0] not in chosen_words]
            if len(candidates) > 0:
                mutation = rng.choice(candidates)
                chosen_words.add(mutation[0])
                mutations.append(mutation)
                group.remove(mutation)
            else:
                current_group_number += 1
        mutations_list.append(mutations)

    return mutations_list


MUTATION_SELECTION_STRATEGIES = {
    "random": _select_mutations_random,
    "most_common_first": _select_mutations_most_common_first,
}


def _get_selection_strategy_func(selection_strategy: str)\
        -> Callable[[List[Word], int, int, Random], List[List[Tuple[Word, str]]]]:
    """
    A basic dictionary lookup to select either of the two mutation selection strategies.
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
    Returns a list of mutated version of the given input sentence. Mutations are based on synonyms
    and hypernyms of nontrivial words. Nontrivial words are words that are not stopwords (as defined
    by WordNet and have at least one synonym/hypernym available. The synonyms/hypernyms are based on
    WordNet Synsets.

    WordNet suggests a lot of variants (synonyms or hypernyms) for each nontrivial word. The same
    variant can be suggested multiple times in different Synsets. This can be taken as a measure of
    'variant quality'. There are two strategies available for choosing variants as mutations:

    1) "random": each variant has an equal chance of being chosen, regardless of how many times it
    was suggested by WordNet.
    2) "most_common_first": variants that were suggested often are chosen first.

            Parameters:
                ``input_sentence`` (``str``): the input sentence

                ``num_replacements`` (``int``): the number of words that should be replaced per
                output sentence

                ``num_variants`` (``int``): the number of output sentences

                ``selection_strategy`` (``str``): the desired selection mutation strategy, which can
                be either "random" (default) or "most_common_first"

                ``random_seed`` (``int``):

            Returns:
                A list mutated sentences.
    """
    # Set thread safe seed for reproducibility
    rng = random_pkg.Random()
    rng.seed(a=random_seed)

    tokens = word_tokenize(input_sentence)
    tokens_with_pos_tags = nltk.pos_tag(tokens)
    words = [Word.from_tuple(t) for t in tokens_with_pos_tags]
    non_trivial_words = {word: index for (index, word) in enumerate(words) if word.is_nontrivial}

    selection_strategy_func = _get_selection_strategy_func(selection_strategy)

    mutations_list = selection_strategy_func(list(non_trivial_words.keys()),
                                             num_replacements,
                                             num_variants,
                                             rng)

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


if __name__ == "__main__":
    test_sentence = "Uploading files via JSON Post request to a Web Service provided by Teambox"
    result = mutate_by_replacement(test_sentence,
                                   num_replacements=2,
                                   num_variants=10,
                                   selection_strategy="most_common_first")
    print(f"ORIGINAL: \n{test_sentence}")
    print("VARIANTS:")
    for x in result:
        print(x)
