import json
import os
import random
from datetime import datetime
import numpy
from deap import base, creator, tools
import plots
from markov import mkv
import deap_ops
import constants


def run_ga(file_in, random_seed, novelty_method):

    # set random seed
    random.seed(random_seed)
    numpy.random.seed(random_seed)

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

    # read input model and form classes
    # for generation and evaluation of individuals
    mfi = "data/models/" + file_in
    if os.path.exists(mfi):
        tps, classes, patterns, gen_sequence_length = mkv.load_model(mfi)
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
    stats["const"]["NOV_FIT_THRESH"] = constants.NOV_FIT_THRESH
    stats["method"] = novelty_method

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
    toolbox.register("dirInd", deap_ops.create_individual)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.dirInd)
    # GA operators
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.3)
    # toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.3, indpb=0.4)
    # selection
    toolbox.register("select", tools.selSPEA2)
    # eval
    # toolbox.register("evaluate", lambda x: (deap_ops.eval_fitness(x, tps, classes, patterns, gen_sequence_length), 0))

    # set novelty function
    # if novelty_method == "phenotype":
    #     toolbox.register("evaluateMulti",
    #                       lambda x: deap_ops.eval_fitness_and_novelty_phenotype(x, tps, classes, patterns,
    #                                                                                archive, gen_sequence_length))
    #     stats["method"] = "eval_fitness_and_novelty_phenotype"
    # elif novelty_method == "phenotype_ncd":
    #     toolbox.register("evaluateMulti",
    #                      lambda x: deap_ops.eval_fitness_and_novelty_phenotype_ncd(x, tps, classes, patterns, pop,
    #                                                                                archive, gen_sequence_length))
    #     stats["method"] = "eval_fitness_and_novelty_phenotype_ncd"
    # else:
    toolbox.register("evaluateMulti", lambda x: deap_ops.eval_fitness_and_novelty_genotype(x, tps, classes, patterns, pop, archive, gen_sequence_length))

    # decorators for normalizing individuals
    toolbox.decorate("mate", deap_ops.normalize_individuals())
    toolbox.decorate("mutate", deap_ops.normalize_individuals())

    # evaluation function: (fitness or fitness-novelty)
    evaluation_function = toolbox.evaluateMulti
    # feasible_individuals = 0
    # create the population
    pop = toolbox.population(n=constants.POP_SIZE)

    # generations
    for g in range(constants.NGEN):

        # new stats page
        stats[g] = dict()

        # novelty search: choose evaluate function (fitness or multi)
        # if novelty_method.find("fitness_only") == -1:
        #     if feasible_individuals >= constants.NOV_T_MAX:
        #         # fitness + novelty
        #         evaluation_function = toolbox.evaluateMulti
        #     elif feasible_individuals <= constants.NOV_T_MIN:
        #         # fitness
        #         evaluation_function = toolbox.evaluate

        ###################################################################

        # EVALUATION
        # t1 = datetime.now()
        # feasible_individuals = 0
        fit_values = list(map(evaluation_function, pop))
        for ind, fit in zip(pop, fit_values):
            ind.fitness.values = fit
            # count feasible individuals for novelty search
            # if fit[0] > constants.NOV_FIT_THRESH:
            #     feasible_individuals = feasible_individuals + 1
        # print("Eval... time: ", (datetime.now() - t1).total_seconds(), "s.")

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
        values = toolbox.map(evaluation_function, invalid_ind)
        for ind, fit in zip(invalid_ind, values):
            ind.fitness.values = fit
        # print("Eval invalid...", "time: " + str((datetime.now() - t2).total_seconds()))

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
        # stats[g]["method"] = "F" if evaluation_function == toolbox.evaluate else "H"
        stats[g]["method"] = "H"
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
        bb_stats[i]["seqs"] = mkv.generate_with_weights(
            tps=tps, weights=bb, n_seq=constants.NUM_SEQS, occ_per_seq=gen_sequence_length, start_pool=classes["sp"]
        )

    # for ind in pop:
    #     # if ind.fitness.values[0] > 0.4:
    #     print(np.round(ind, 2), "->", ind.fitness.values)

    print("time elapsed :", stats["time"], "sec.")

    # save generated sequences
    with open(dir_out + "generated.json", "w") as fp:
        json.dump(bb_stats, fp, default=mkv.serialize_sets)

    # save stats
    with open(dir_out + "stats.json", "w") as fp:
        json.dump(stats, fp, default=mkv.serialize_sets)

    # plots.plot_fits(dir_out, constants.NGEN, fits, novs, stats["method"])
    plots.plot_data(dir_out, constants.NGEN, fits, novs, arch_s, stats["method"])
    plots.plot_pareto(dir_out, pop_plot, best_plot, stats["method"])


if __name__ == "__main__":
    run_ga("input_43_0.75_1.0_3_3", 43, "multi_log")
