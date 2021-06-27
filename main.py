import json
import os
import random
from datetime import datetime
import numpy as np
from deap import base, creator, tools
import plots
import markov
import deap_ops
import constants


def run_ga(file_in, random_seed, novelty_method):

    # set random seed
    # https://numpy.org/doc/1.18/reference/random/parallel.html
    random.seed(random_seed)
    rng = np.random.default_rng(random_seed)
    # numpy.random.seed(random_seed)

    root_out = "data/out/" + file_in + "/"
    dir_out = root_out + novelty_method + "_" + datetime.now().strftime("%Y%m%d-%H.%M.%S") + "/"

    # Create target dir if don't exist
    if not os.path.exists(root_out):
        os.mkdir(root_out)
    # Create dir_out if don't exist
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    else:
        print("Directory ", dir_out, "already exists")

    # read input model
    # for generation and evaluation of individuals
    mfi = "data/models/" + file_in
    if os.path.exists(mfi):
        tps, start_pool, gen_sequence_length = markov.load_model(mfi)
    else:
        print("ERROR: no model dir")
        return 0

    # time
    start_time = datetime.now()

    # init archive
    archive = []

    # STATS
    stats = dict()
    stats["const"] = dict()
    stats["const"]["file_in"] = file_in
    stats["const"]["gen_sequence_length"] = gen_sequence_length
    stats["const"]["NGEN"] = constants.NGEN
    stats["const"]["POP_SIZE"] = constants.POP_SIZE
    stats["const"]["N_ELITE"] = constants.N_ELITE
    stats["const"]["NOV_T_MIN"] = constants.NOV_T_MIN
    stats["const"]["NOV_T_MAX"] = constants.NOV_T_MAX

    # for plot
    fits = []
    novs = []
    arch_s = []

    # DEAP
    # toolbox
    toolbox = base.Toolbox()
    # init DEAP fitness and individual for tournament in novelty search
    if not hasattr(creator, "FitnessMaxTN"):
        creator.create("FitnessMaxTN", base.Fitness, weights=(-1.0,))
        creator.create("IndividualTN", list, fitness=creator.FitnessMaxTN)
    # init DEAP fitness and individual
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(-1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox.register("dirInd", lambda: deap_ops.create_individual(rng))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.dirInd)
    # GA operators
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.3)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.3, indpb=0.4)
    # selection
    toolbox.register("select", tools.selSPEA2)

    # set objective
    if novelty_method == "multi_log_switch":
        toolbox.register("evaluateMulti",
                         lambda x: deap_ops.eval_fitness_and_novelty_log_switches(x, tps, start_pool, pop, archive,
                                                                                  gen_sequence_length))
    else:
        toolbox.register("evaluateMulti",
                         lambda x: deap_ops.eval_fitness_and_novelty_log_min(x, tps, start_pool, pop, archive,
                                                                             gen_sequence_length))

    # decorators for normalizing individuals
    toolbox.decorate("mate", deap_ops.normalize_individuals())
    toolbox.decorate("mutate", deap_ops.normalize_individuals())

    # create the population
    pop = toolbox.population(n=constants.POP_SIZE)

    # generations
    for g in range(constants.NGEN):

        # new stats page
        stats[g] = dict()

        # EVALUATION
        # t1 = datetime.now()
        # feasible_individuals = 0
        fit_values = list(map(toolbox.evaluateMulti, pop))
        for ind, fit in zip(pop, fit_values):
            ind.fitness.values = fit

        # SELECTION
        offspring = list(map(toolbox.clone, toolbox.select(pop, k=constants.POP_SIZE - constants.N_ELITE)))
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
        # t2 = datetime.now()
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        values = toolbox.map(toolbox.evaluateMulti, invalid_ind)
        for ind, fit in zip(invalid_ind, values):
            ind.fitness.values = fit

        # new pop
        pop[:] = elite + offspring
        # delete archive duplicates entries
        archive = list(set(archive))

        ###################################################################
        # SAVE STATISTICS

        res = [ind.fitness.values for ind in pop]
        fits.append(sum(x[0] for x in res) / constants.POP_SIZE)
        novs.append(sum(x[1] for x in res) / constants.POP_SIZE)
        arch_s.append(len(archive))

        # save stats
        # in case use copy.deepcopy()
        stats[g]["pop"] = pop[:]
        stats[g]["fitness"] = res[:]
        stats[g]["archive"] = archive[:]

    # end ga

    ###############################################################
    #                   OUT, PLOTS and GRAPHS
    ###############################################################
    stats["time"] = (datetime.now() - start_time).total_seconds()

    pop_plot = {"fits": [], "novs": []}
    best_plot = {"fits":[], "novs":[]}
    bb_stats = dict()

    for pb in pop:
        pop_plot["fits"].append(pb.fitness.values[0])
        pop_plot["novs"].append(pb.fitness.values[1])

    bests = toolbox.select(pop, k=7)
    for i,bb in enumerate(bests):
        bb_stats[i] = dict()
        bb_stats[i]["individual"] = bb
        bb_stats[i]["fit"] = bb.fitness.values
        best_plot["fits"].append(bb.fitness.values[0])
        best_plot["novs"].append(bb.fitness.values[1])
        bb_stats[i]["seqs"] = markov.generate_with_weights(
            tps=tps, weights=bb, n_seq=constants.NUM_SEQS, occ_per_seq=gen_sequence_length, start_pool=start_pool)

    print("time elapsed :", stats["time"], "sec.")

    # save generated sequences
    with open(dir_out + "generated.json", "w") as fp:
        json.dump(bb_stats, fp, default=markov.serialize_sets)

    # result for weights progress
    with open(dir_out + "pop.json", "w") as fp:
        json.dump(pop[0::10], fp, default=markov.serialize_sets)
    # save stats
    with open(dir_out + "stats.json", "w") as fp:
        json.dump(stats, fp, default=markov.serialize_sets)

    # plots.plot_fits(dir_out, constants.NGEN, fits, novs, novelty_method)
    plots.plot_data(dir_out, constants.NGEN, fits, novs, arch_s, novelty_method)
    plots.plot_pareto(dir_out, pop_plot, best_plot, novelty_method)


if __name__ == "__main__":
    # run_ga("input", 43, "multi_log_switches")
    run_ga("all_songs_in_G", 43, "multi_log_min")
