import os
import random
from datetime import datetime
from shutil import copyfile
import bcolors as bc
import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools
import utils
from markov import mkv
import deap_ops
import constants

###############################################################
#               init, input, output
###############################################################
# calculate model and form classes for generation and evaluation of individuals

file_in = "data/input2.txt"
dir_out = "data/out/results_" + datetime.now().strftime("%Y%m%d-%H.%M.%S") + "/"
os.mkdir(dir_out)
copyfile(file_in, dir_out + "input.txt")

# read
sequences = utils.read_from_file(file_in, "")
# model
tps, classes, patterns = mkv.compute(sequences, dir_out + "model/")
start_time = datetime.now()

# DEAP
# toolbox
toolbox = base.Toolbox()
# init DEAP fitness and individual for tournament in novelty search
creator.create("FitnessMaxTN", base.Fitness, weights=(1.0,))
creator.create("IndividualTN", list, fitness=creator.FitnessMaxTN)
# init DEAP fitness and individual
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("dirInd", deap_ops.create_individual)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.dirInd)
# GA operators
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", tools.mutUniformInt, low=0, up=100, indpb=0.3)
toolbox.register("mutate", tools.mutGaussian, mu=1, sigma=1, indpb=0.3)
# decorators for normalizing individuals
toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("selectTournament", tools.selTournament, k=constants.POP_SIZE / 10, tournsize=5)
toolbox.decorate("mate", deap_ops.normalize_individuals())
toolbox.decorate("mutate", deap_ops.normalize_individuals())
# only fitness
toolbox.register("evaluate", lambda x: (deap_ops.eval_fitness(x, tps, classes, patterns), 0))
# fitness + novelty
toolbox.register("evaluateMulti", lambda x: deap_ops.eval_fitness_and_novelty(x, tps, classes, patterns, pop, archive))
toolbox.register("selectspea2", tools.selSPEA2, k=constants.N_ELITE)

# set random seed
random.seed(7)
# init archive
archive = []
# create the population
pop = toolbox.population(n=constants.POP_SIZE)

# evaluation function: (fitness or fitness-novelty)
evaluation_function = toolbox.evaluate
feasible_individuals = 0

fits = []
novs = []
arch_s = []
for g in range(constants.NGEN):

    # novelty search: choose evaluate function (fitness or multi)
    if feasible_individuals >= constants.NOV_T_MAX:
        # fitness + novelty
        evaluation_function = toolbox.evaluateMulti
    elif feasible_individuals <= constants.NOV_T_MIN:
        # fitness
        evaluation_function = toolbox.evaluate

    ###################################################################
    # EVALUATION
    ###################################################################
    feasible_individuals = 0
    fit_values = list(map(evaluation_function, pop))

    for ind, fit in zip(pop, fit_values):
        ind.fitness.values = fit
        # count feasible individuals for novelty search
        if fit[0] > constants.NOV_FIT_THRESH:
            feasible_individuals = feasible_individuals + 1

    # print used method
    if evaluation_function == toolbox.evaluate:
        print(g,": ", bc.PASS + "F" + bc.ENDC," - ", feasible_individuals, " - ", len(archive))
    elif evaluation_function == toolbox.evaluateMulti:
        print(g,": ", bc.BLUE + "H" + bc.ENDC," - ", feasible_individuals, " - ", len(archive))
    else:
        print("FATAL ERROR: NO METHOD FOUND")

    ###################################################################
    # SELECTION
    ###################################################################
    # offspring = toolbox.select(pop, len(pop))  # A. DO tournament
    parents = toolbox.selectspea2(pop)  # B. Select the elite

    # offspring = list(map(toolbox.clone, offspring))  # A. Clone the selected individuals
    offspring = list(map(toolbox.clone, parents))  # B. Clone parents
    ###################################################################
    # CROSSOVER
    ###################################################################
    # A. entire pop
    # for child1, child2 in zip(offspring[::2], offspring[1::2]):
    #     if random.random() < constants.CXPB:
    #         toolbox.mate(child1, child2)
    #         del child1.fitness.values
    #         del child2.fitness.values

    # B. new individuals of the population
    new = []
    i = 0
    for child1 in offspring:
        for child2 in offspring:
            if i < (constants.POP_SIZE - constants.N_ELITE):
                child1_copy = toolbox.clone(child1)
                child2_copy = toolbox.clone(child2)
                a, b = toolbox.mate(child1_copy, child2_copy)
                new.append(a)
                new.append(b)
                i = i + 2

    ###################################################################
    # MUTATION
    ###################################################################
    for mutant in offspring:
        if random.random() < constants.MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    values = toolbox.map(evaluation_function, invalid_ind)
    for ind, fit in zip(invalid_ind, values):
        ind.fitness.values = fit

    # new pop
    # pop[:] = offspring  # A. The population is entirely replaced by the offspring
    pop = parents + new  # B. create the new pop (tot = 10 + 90)

    ###################################################################
    # STATISTICS
    ###################################################################
    res = [ind.fitness.values for ind in pop]
    fits.append(sum(x[0] for x in res) / constants.POP_SIZE)
    novs.append(sum(x[1] for x in res) / constants.POP_SIZE)
    arch_s.append(len(archive))

###############################################################
#                   OUT, PLOTS and GRAPHS
###############################################################
print("time elapsed :", (datetime.now() - start_time).total_seconds(), "sec.")
plt.xlabel('ngen')
plt.yticks(np.linspace(0, 1, 11))
plt.plot(range(0, constants.NGEN), fits, label="fitness")
plt.plot(range(0, constants.NGEN), novs, label="novelty")
plt.legend()
plt.tight_layout()
plt.draw()
plt.savefig(dir_out+"graph")

for ind in pop:
    # if ind.fitness.values[0] > 0.4:
    print(np.round(ind, 2), " -> ", ind.fitness.values)
