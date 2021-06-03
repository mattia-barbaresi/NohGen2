import json
import os
import random
from datetime import datetime
from shutil import copyfile
import bcolors as bc
import numpy as np
from deap import base, creator, tools
import plots
import utils
from Params import Params
from markov import mkv
import deap_ops
import constants


def run_ga(params):
    # set random seed
    random.seed(params.random_seed)
    # time
    start_time = datetime.now()
    # input, output
    dir_out, file_in = _create_out_dir(params)
    # calculate model and form classes for generation and evaluation of individuals
    sequences = utils.read_from_file(file_in, params.file_in["sep"])
    fc_model = mkv.compute(sequences, dir_out + "model/")
    tps, tps_s, classes, patterns = fc_model

    # init archive
    archive = []

    # STATS
    stats = dict()
    stats["const"] = dict()
    stats["const"]["file_in"] = file_in
    stats["const"]["NGEN"] = constants.NGEN
    stats["const"]["POP_SIZE"] = constants.POP_SIZE
    stats["const"]["N_ELITE"] = constants.N_ELITE
    stats["const"]["NOV_T_MIN"] = constants.NOV_T_MIN
    stats["const"]["NOV_T_MAX"] = constants.NOV_T_MAX
    stats["const"]["NOV_FIT_THRESH"] = constants.NOV_FIT_THRESH
    stats["const"]["NOV_ARCH_MIN_DISS"] = constants.NOV_ARCH_MIN_DISS
    # for plot
    fits = []
    novs = []
    arch_s = []

    # DEAP
    # toolbox
    toolbox = base.Toolbox()
    # init DEAP fitness and individual for tournament in novelty search
    if not hasattr(creator, "FitnessMaxTN"):
        creator.create("FitnessMaxTN", base.Fitness, weights=(1.0,))
        creator.create("IndividualTN", list, fitness=creator.FitnessMaxTN)
    # init DEAP fitness and individual
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox.register("dirInd", deap_ops.create_individual)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.dirInd)
    # GA operators
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.3)
    # toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0.5, sigma=0.5, indpb=0.2)
    # selection
    toolbox.register("selectspea2", tools.selSPEA2)
    # eval
    toolbox.register("evaluate", lambda x: (deap_ops.eval_fitness(x, tps, classes, patterns), 0))

    # set novelty function
    if params.novelty_method == "phenotype":
        toolbox.register("evaluateMulti",
                         lambda x: deap_ops.eval_fitness_and_novelty_phenotype(x, tps, classes, patterns, archive))
        stats["method"] = "eval_fitness_and_novelty_phenotype"
    elif params.novelty_method == "phenotype_ncd":
        toolbox.register("evaluateMulti",
                         lambda x: deap_ops.eval_fitness_and_novelty_phenotype_ncd(x, tps, classes, patterns, pop, archive))
        stats["method"] = "eval_fitness_and_novelty_phenotype_ncd"
    else:
        toolbox.register("evaluateMulti",
                         lambda x: deap_ops.eval_fitness_and_novelty_genotype(x, tps, classes, patterns, pop, archive))
        stats["method"] = "eval_fitness_and_novelty_genotype"

    # decorators for normalizing individuals
    toolbox.decorate("mate", deap_ops.normalize_individuals())
    toolbox.decorate("mutate", deap_ops.normalize_individuals())

    # evaluation function: (fitness or fitness-novelty)
    evaluation_function = toolbox.evaluate
    feasible_individuals = 0
    # create the population
    pop = toolbox.population(n=constants.POP_SIZE)

    # generations
    for g in range(constants.NGEN):

        # new stats page
        stats[g] = dict()

        # novelty search: choose evaluate function (fitness or multi)
        if feasible_individuals >= constants.NOV_T_MAX:
            # fitness + novelty
            evaluation_function = toolbox.evaluateMulti
        elif feasible_individuals <= constants.NOV_T_MIN:
            # fitness
            evaluation_function = toolbox.evaluate

        ###################################################################

        # EVALUATION
        feasible_individuals = 0
        fit_values = list(map(evaluation_function, pop))
        for ind, fit in zip(pop, fit_values):
            ind.fitness.values = fit
            # count feasible individuals for novelty search
            if fit[0] > constants.NOV_FIT_THRESH:
                feasible_individuals = feasible_individuals + 1

        # SELECTION
        offspring = list(map(toolbox.clone, toolbox.selectspea2(pop, k=constants.POP_SIZE - constants.N_ELITE)))
        elite = list(map(toolbox.clone, offspring[:constants.N_ELITE]))  # Select the elite

        random.shuffle(offspring)

        # CROSSOVER
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < constants.CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # MUTATION
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
        pop[:] = elite + offspring

        ###################################################################
        # SAVE STATISTICS

        # print used method
        # if evaluation_function == toolbox.evaluate:
        #     print(g, ":", bc.PASS + "F" + bc.ENDC, "fi=" + str(feasible_individuals), "a=" + str(len(archive)))
        # elif evaluation_function == toolbox.evaluateMulti:
        #     print(g, ":", bc.BLUE + "H" + bc.ENDC, "fi=" + str(feasible_individuals), "a=" + str(len(archive)))
        # else:
        #     print("FATAL ERROR: NO METHOD FOUND")

        res = [ind.fitness.values for ind in pop]
        fits.append(sum(x[0] for x in res) / constants.POP_SIZE)
        novs.append(sum(x[1] for x in res) / constants.POP_SIZE)
        arch_s.append(len(archive))

        # save stats
        # in case use copy.deepcopy()
        stats[g]["method"] = "F" if evaluation_function == toolbox.evaluate else "H"
        stats[g]["pop"] = pop[:]
        stats[g]["fitness"] = res[:]
        stats[g]["archive"] = archive[:]

    ###############################################################
    #                   OUT, PLOTS and GRAPHS
    ###############################################################
    stats["time"] = (datetime.now() - start_time).total_seconds()
    # print("time elapsed :", stats["time"], "sec.")

    # for ind in pop:
    #     # if ind.fitness.values[0] > 0.4:
    #     print(np.round(ind, 2), "->", ind.fitness.values)

    # save stats
    with open(dir_out + "stats.json", "w") as fp:
        json.dump(stats, fp, default=mkv.serialize_sets)

    plots.plot_tps(dir_out, tps_s)
    plots.plot_fits(dir_out, constants.NGEN, fits, novs, stats["method"])
    plots.plot_data(dir_out, constants.NGEN, fits, novs, arch_s, stats["method"])


# creates output dirs and files
def _create_out_dir(params):
    file_in = "data/" + params.file_in["name"] + ".txt"
    root_out = "data/out/" + params.novelty_method + "_" + datetime.now().strftime("%Y%m%d-%H.%M.%S") + "/"
    dir_out = root_out + params.file_in["name"] + "_" + str(params.random_seed) + "/"

    # Create target dir if don't exist
    if not os.path.exists(root_out):
        os.mkdir(root_out)
    # Create dir_out if don't exist
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    else:
        print("Directory ", dir_out, "already exists")
    copyfile(file_in, dir_out + params.file_in["name"] + ".txt")
    return dir_out, file_in


if __name__ == "__main__":
    run_ga(Params(file_in={"name":"all_songs_in_G","sep":""}, random_seed=7, novelty_method="genotype"))
