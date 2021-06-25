import random

import constants
import numpy as np
import novelty_search
import markov


# fun for creating an individual
def create_individual(rng):
    # using dirichlet distribution
    # v = [random.random() for _ in range(constants.IND_SIZE)]
    # sm = sum(v)
    # ind = list([x/sm for x in v])
    v = rng.dirichlet(np.ones(constants.IND_SIZE), size=None)
    return v


# generate sequences from given individual(weights)
def gen_sequences(individual, tps, start_pool, gen_seq_len):
    return markov.generate_with_weights(
        tps=tps, weights=individual, n_seq=constants.NUM_SEQS, occ_per_seq=gen_seq_len, start_pool=start_pool)


# on genotype
def eval_fitness_and_novelty_log_min(individual, tps, start_pool, population, archive, gen_seq_len):

    sequences = gen_sequences(individual, tps, start_pool, gen_seq_len)
    res = markov.sequences_markov_support_log(sequences, tps)
    fit = sum(res) / constants.NUM_SEQS

    novelty_search.archive_assessment(individual, archive)
    nov = novelty_search.novelty(individual, population, archive)
    return fit, nov


# on genotype
def eval_fitness_and_novelty_log_switches(individual, tps, start_pool, population, archive, gen_seq_len):
    sequences = gen_sequences(individual, tps, start_pool, gen_seq_len)
    res = markov.sequences_markov_support_with_switches(sequences, tps, [1, 1, 1, 1, 1, 1])
    fit = sum(res) / constants.NUM_SEQS

    novelty_search.archive_assessment(individual, archive)
    nov = novelty_search.novelty(individual, population, archive)
    return fit, nov


# decorator to normalize individuals
def normalize_individuals():
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                mn = min(child)
                if mn < 0:
                    child[:] = list([i - mn for i in child])
                sm = sum(child)
                if sm == 0:
                    sz = len(child)
                    child[:] = list([1/sz for _ in range(sz)])
                else:
                    for i in range(len(child)):
                        child[i] = child[i]/sm
            return offspring
        return wrapper
    return decorator
