import numpy as np
from deap import creator, tools
import constants
import metrics


# individual dissimilarity respect to the archive
# avg(diss_fun) of individual from MAX_ARCH-most similar exemplars in archive
def archive_dissim(individual, archive, dissimil_fun=metrics.norm_dissimilarity):
    values = []
    for x in archive:
        if np.array_equal(individual,x):
            return 0
        values.append(dissimil_fun(x, individual))
    dissimilarity = 0
    values.sort()
    # select most similar k neighbours (= min dissimilarity)
    max_len = min(constants.MAX_ARCH, len(archive))
    for i in range(0, max_len):
        dissimilarity = dissimilarity + values[i]
    return dissimilarity / max_len


# return novelty value of the individual (default = on genotype)
# calculated as the dissimilarity from the 4 most similar neighbours from (pop U archive)
def novelty(individual, population, archive, dissimil_fun=metrics.norm_dissimilarity, simil_fun=metrics.norm_similarity):
    if len(archive) == 0:
        print("- archive with 0 entries!")

    # select the neighbours (pop + archive)
    pop_selected = select(population, individual, archive, simil_fun)
    nov = 0
    # calculate individual dissimilarity (novelty)
    for x in pop_selected:
        nov = nov + dissimil_fun(individual, x)
    nov = nov / len(pop_selected)
    return nov


def archive_assessment(individual, archive, dissim_fun=metrics.norm_dissimilarity):
    arch_len = len(archive)
    # conditions needed to add the individual to the archive
    if arch_len == 0 or archive_dissim(individual, archive, dissimil_fun=dissim_fun) > constants.NOV_ARCH_MIN_DISS:
        archive.append(tuple(individual))


def archive_assessment_bestInPop(population, archive, dissim_fun=metrics.norm_dissimilarity):
    arch_len = len(archive)
    for individual in population:
        # conditions needed to add the individual to the archive
        if arch_len == 0 or archive_dissim(individual, archive, dissimil_fun=dissim_fun) > constants.NOV_ARCH_MIN_DISS:
            archive.append(tuple(individual))

########################################################################
# novelty on phenotype
# def novelty_phenotype(gseqs, archive):
#     if len(archive) == 0:
#         print("- archive with 0 entries!")
#
#     # select the neighbours
#     pop_selected = select(population, individual, archive)
#     value = 0
#     # calculate individual dissimilarity (novelty)
#     for x in pop_selected:
#         value = value + ind_dissimilarity(individual, x)
#     value = value / len(pop_selected)
#     return value
#
# def archive_assessment_phenotype():

########################################################################


def create_individuals(population, individual_to_compute_novelty, similarity_fun):
    new_population = []
    for individual_in_population in population:
        if not np.array_equal(individual_to_compute_novelty, individual_in_population):
            new_individual = creator.IndividualTN(individual_in_population)
            new_individual.fitness.values = similarity_fun(individual_to_compute_novelty, new_individual),
            new_population.append(new_individual)
    return new_population


def select(population, individual_to_compute_novelty, archive, simil_fun):
    # select the most similar to the individual
    new_population = create_individuals(population, individual_to_compute_novelty, simil_fun)
    similar = tools.selTournament(new_population, k=4, tournsize=5)

    # calculate similarity to individuals in archive
    arch = create_individuals(archive, individual_to_compute_novelty, simil_fun)

    # the whole population is composed by selected individuals and archive
    pop_resulting = similar + arch

    # select the 4 most similar neighbours
    ind_selected = tools.selBest(pop_resulting, 4)

    return ind_selected
