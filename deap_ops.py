import bz2

import constants
import numpy as np
import novelty_search
from markov import mkv, fc


# NCD from NohGenerator
def compute_ncd(a, b):
    ca = float(len(bz2.compress(a)))
    cb = float(len(bz2.compress(b)))
    cab = float(len(bz2.compress(a + b)))
    return (cab - min(ca, cb)) / max(ca, cb)


# fun for creating an individual
def create_individual():
    # using dirichlet distribution
    v = np.random.default_rng().dirichlet(np.ones(constants.IND_SIZE), size=None)
    # for floating error
    # v[-1] = v[-1] + (1.0 - sum(v))
    return v


# fun for evaluating individuals
def eval_fitness(individual, tps, classes, patterns):
    """
    Generates 100 sequences with given tps-model and evaluate each sequences with given classes and patterns.
    Return percentage of hits.
    """

    sequences = mkv.generate_with_weights(
        tps=tps, weights=individual, n_seq=constants.NUM_SEQS, occ_per_seq=constants.SEQUENCE_LENGTH,
        start_pool=classes["sp"])
    res = fc.evaluate_sequences(sequences, classes["fc"], patterns)
    return sum(res) / constants.NUM_SEQS


def eval_fitness_and_novelty(individual, tps, classes, patterns, population, archive):
    fit = eval_fitness(individual, tps, classes, patterns)
    novelty_search.archive_assessment(individual, fit, archive)
    nov = novelty_search.novelty(individual, population, archive)
    return fit, nov


# decorator to normalize individuals
def normalize_individuals():
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                sm = sum(child)
                for i in range(len(child)):
                    child[i] = child[i]/sm
            return offspring
        return wrapper
    return decorator
