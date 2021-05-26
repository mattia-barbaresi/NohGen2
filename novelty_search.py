import bz2
import numpy as np
from deap import creator, tools
import constants


# similar = 1, dissimilar = 0
def ind_similarity(ind1, ind2):
    # using euclidean distance
    a = np.array(ind1)
    b = np.array(ind2)
    # if a is equal b euclidean norm of (a-b) is 0
    return 1 - np.sum((a - b) ** 2) ** 0.5


# similar = 0, dissimilar = 1
def ind_dissimilarity(ind1, ind2):
    return 1 - ind_similarity(ind1, ind2)


def archive_dissim(individual, archive):
    len_a = len(archive)
    # if no entries in archive then add the individual
    if len_a == 0:
        return "-"
    else:
        values = []
        for x in archive:
            values.append(ind_dissimilarity(x, individual))
        # select the most similar images (min dissimilarity)
        values.sort()
        max_len = min(constants.MAX_ARCH, len_a)
        dissimilarity = 0
        for i in range(0, max_len):
            dissimilarity = dissimilarity + values[i]
        return dissimilarity / max_len


# return novelty value of the individual
# calculated as the dissimilarity from the 4 most similar neighbours from (pop U archive)
def novelty(individual, population, archive):
    if len(archive) == 0:
        print("- archive with 0 entries!")

    # select the neighbours
    pop_selected = select(population, individual, archive)
    value = 0
    # calculate individual dissimilarity (novelty)
    for x in pop_selected:
        value = value + ind_dissimilarity(individual, x)
    value = value / len(pop_selected)
    return value


def archive_assessment(individual, evaluation, archive):
    arch_len = len(archive)
    # conditions needed to add the individual to the archive
    if evaluation > constants.NOV_FIT_THRESH:
        # if the archive has no entries or if the dissimilarity between the
        # element and the choreographies in the archive is higher than a threshold
        arch_dissim = archive_dissim(individual, archive)
        if arch_len == 0 or arch_dissim > constants.NOV_ARCH_MIN_DISS:
            archive.append(individual)


def create_individuals(population, individual_to_compute_novelty, is_archive):
    new_population = []
    first = True
    for individual_in_population in population:
        # the individual is excluded from the population (is not excluded if there is more than one copy of it
        # the individual is not excluded from the archive
        if not np.array_equal(individual_to_compute_novelty, individual_in_population) or \
                (not first) or is_archive:
            new_individual = creator.IndividualTN(individual_in_population)
            new_individual.fitness.values = ind_similarity(individual_to_compute_novelty, new_individual),
            new_population.append(new_individual)
        else:
            first = False
    return new_population


def select(population, individual_to_compute_novelty, archive):
    # select the most similar to the individual
    new_population = create_individuals(population, individual_to_compute_novelty, False)
    pop_selected = tools.selTournament(new_population, k=4, tournsize=5)

    # calculate similarity to individuals in archive
    arch = create_individuals(archive, individual_to_compute_novelty, True)

    # the whole population is composed by selected individuals and archive
    pop_resulting = pop_selected + arch

    # select the 4 most similar neighbours
    ind_selected = tools.selBest(pop_resulting, 4, )

    return ind_selected
