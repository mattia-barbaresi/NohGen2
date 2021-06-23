import math

import constants
import numpy as np
import metrics
import novelty_search
from markov import mkv, fc


# similar = 1, dissimilar = 0
def ncd_similarity(ind1, ind2, tps, classes, gen_seq_len):
    s1 = mkv.generate_with_weights(
        tps=tps, weights=ind1, n_seq=constants.NUM_SEQS, occ_per_seq=gen_seq_len,
        start_pool=classes["sp"])
    s2 = mkv.generate_with_weights(
        tps=tps, weights=ind2, n_seq=constants.NUM_SEQS, occ_per_seq=gen_seq_len,
        start_pool=classes["sp"])
    return metrics.compute_ncd(''.join(s1), ''.join(s2))


# similar = 0, dissimilar = 1
def ncd_dissimilarity(ind1, ind2, tps, classes, gen_seq_len):
    return 1 - ncd_similarity(ind1, ind2, tps, classes, gen_seq_len)


# fun for creating an individual
def create_individual():
    # using dirichlet distribution
    v = np.random.default_rng().dirichlet(np.ones(constants.IND_SIZE), size=None)
    # for floating error
    # v[-1] = v[-1] + (1.0 - sum(v))
    return v


# generate sequences from given individual(weights)
def gen_sequences(individual, tps, classes, gen_seq_len):
    return mkv.generate_with_weights(
        tps=tps, weights=individual, n_seq=constants.NUM_SEQS, occ_per_seq=gen_seq_len, start_pool=classes["sp"])


# fun for evaluating individuals
def eval_fitness(individual, tps, classes, patterns, gen_seq_len):
    """
    Generates NUM_SEQS sequences with given tps-model and evaluate each sequences with given classes and patterns.
    Return percentage of hits.
    """
    sequences = gen_sequences(individual, tps, classes, gen_seq_len)
    # use similarity instead of perfect match
    # res = fc.evaluate_sequences2(sequences, classes["fc"], patterns)
    res = mkv.sequences_markov_support_log(sequences, tps)

    return sum(res) / constants.NUM_SEQS


# on genotype
def eval_fitness_and_novelty_genotype(individual, tps, classes, patterns, population, archive, gen_seq_len):
    fit = eval_fitness(individual, tps, classes, patterns, gen_seq_len)
    novelty_search.archive_assessment(individual, fit, archive)
    nov = novelty_search.novelty(individual, population, archive)
    return fit, nov


# use ncd instead of novelty on phenotype
def eval_fitness_and_novelty_phenotype_ncd(individual, tps, classes, patterns, population, archive, gen_seq_len):
    fit = eval_fitness(individual, tps, classes, patterns, gen_seq_len)
    novelty_search.archive_assessment(individual, fit, archive,
                                      dissim_fun=(lambda x,y: ncd_dissimilarity(x,y,tps, classes, gen_seq_len)))
    nov = novelty_search.novelty(individual, population, archive,
                                 simil_fun=lambda x,y: ncd_similarity(x,y,tps,classes),
                                 dissimil_fun=(lambda x,y: ncd_dissimilarity(x,y,tps,classes, gen_seq_len)))
    return fit, nov


# use novelty_search in phenotype = on generated sequences (no population, string similarity)
def eval_fitness_and_novelty_phenotype(individual, tps, classes, patterns, archive, gen_seq_len):
    sequences = gen_sequences(individual, tps, classes, gen_seq_len)
    # fitness
    res = fc.evaluate_sequences2(sequences, classes["fc"], patterns)
    fit = sum(res) / constants.NUM_SEQS
    # novelty
    nov = 0
    if fit > constants.NOV_FIT_THRESH:
        for seq in sequences:
            if len(archive) == 0 or novelty_search.archive_dissim(seq, archive, metrics.str_dissimilarity) > constants.NOV_ARCH_MIN_DISS:
                archive.append(seq)
            nov += novelty_search.novelty(seq, sequences, archive, dissimil_fun=metrics.str_dissimilarity,
                                          simil_fun=metrics.str_similarity)
    # avg novelty
    nov = nov / constants.NUM_SEQS

    return fit, nov


# decorator to normalize individuals
def normalize_individuals():
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                mn = min(child)
                if mn < 0:
                    child[:] = [i - mn for i in child]
                sm = sum(child)
                for i in range(len(child)):
                    child[i] = child[i]/sm
            return offspring
        return wrapper
    return decorator
