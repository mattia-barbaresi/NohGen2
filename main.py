"""
Runs a GA: evolves weights used to modulate generation of sequences using the given markov TPs model
"""
import json
import os
import random
from datetime import datetime
import numpy as np
from deap import base, creator, tools
import novelty_search
import plots
import markov
import deap_ops
import constants
import statistics


def run_ga(file_in, random_seed, novelty_method):
    # set random seed
    # https://numpy.org/doc/1.18/reference/random/parallel.html
    random.seed(random_seed)
    rng = np.random.default_rng(random_seed)

    root_out = "data/out/" + file_in + "_" + str(random_seed) + "/"
    dir_out = root_out + novelty_method + "_" + datetime.now().strftime("%Y%m%d-%H.%M.%S") + "/"

    # print("starting exec. output dir: ", dir_out)

    # Create target dir if don't exist
    try:
        if not os.path.exists(root_out):
            os.mkdir(root_out)
        # Create dir_out if don't exist
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)
    except OSError as e:
        print("Directory already exists: ", e)

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
    stats["time"] = 0.0
    stats["const"] = dict()
    stats["const"]["file_in"] = file_in
    stats["const"]["gen_sequence_length"] = gen_sequence_length
    stats["const"]["NGEN"] = constants.NGEN
    stats["const"]["POP_SIZE"] = constants.POP_SIZE
    stats["const"]["N_ELITE"] = constants.N_ELITE
    stats["const"]["NOV_ARCH_MIN_DISS"] = constants.NOV_ARCH_MIN_DISS
    stats["const"]["CXPB"] = constants.CXPB
    stats["const"]["MUTPB"] = constants.MUTPB
    stats["const"]["NUM_SEQS"] = constants.NUM_SEQS
    stats["const"]["MAX_FIT_TIMES"] = constants.MAX_FIT_TIMES
    # stats["const"]["NOV_OFFSET"] = constants.NOV_OFFSET
    stats["const"]["NOV_OFFSET"] = "stdev/2"
    stats["final_archive"] = []

    # for plot
    fits = []
    novs = []
    mins = []
    maxs = []
    arch_s = []
    avg_last = 0
    stdev_last = 0

    # DEAP
    # toolbox
    toolbox = base.Toolbox()
    # init DEAP fitness and individual for tournament in novelty search: select best similarity (max)
    if not hasattr(creator, "FitnessMaxTN"):
        creator.create("FitnessMaxTN", base.Fitness, weights=(1.0,))
        creator.create("IndividualTN", list, fitness=creator.FitnessMaxTN)
    # init DEAP fitness and individual
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox.register("dirInd", lambda: deap_ops.create_individual(rng))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.dirInd)
    # GA operators
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.35)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.3, indpb=0.5)
    # selection
    toolbox.register("select", tools.selSPEA2)
    # set objective
    toolbox.register("evaluate",
                     lambda x: deap_ops.eval_fitness(x, tps, start_pool, gen_sequence_length))
    toolbox.register("evaluateMulti",
                     lambda x: deap_ops.eval_fitness_and_novelty(x, tps, start_pool, pop, archive, gen_sequence_length))

    # decorators for normalizing individuals
    toolbox.decorate("mate", deap_ops.normalize_individuals())
    toolbox.decorate("mutate", deap_ops.normalize_individuals())

    # create the population
    pop = toolbox.population(n=constants.POP_SIZE)

    # starts with fitness
    eval_function = toolbox.evaluate
    fit_best = 0
    max_times = constants.MAX_FIT_TIMES
    fit_last = 0

    # generations
    for g in range(constants.NGEN):

        # new stats page
        stats[g] = dict()

        # EVALUATION
        fit_values = list(map(eval_function, pop))
        for ind, fit in zip(pop, fit_values):
            ind.fitness.values = fit

        # SELECTION
        elite = list(map(toolbox.clone, toolbox.select(pop, k=constants.N_ELITE)))
        offspring = list(map(toolbox.clone, toolbox.select(pop, k=constants.POP_SIZE - constants.N_ELITE)))

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
        values = toolbox.map(eval_function, invalid_ind)
        for ind, fit in zip(invalid_ind, values):
            ind.fitness.values = fit

        # SWITCH FUNCTION MIN
        # fit_prev = fit_best
        # # selects the fittest individual in pop (minimum for minimization)
        # fit_best = tools.selBest(pop,k=1)[0].fitness.values[0]
        # if g > 0:
        #     if eval_function == toolbox.evaluate:
        #         if 0.95 <= fit_best/fit_prev <= 1.05:  # if they are similar
        #             max_times -= 1
        #             # switch to multi with novelty
        #             if max_times == 0:
        #                 max_times = constants.MAX_FIT_TIMES
        #                 fit_last = fit_best  # last fit calculated with only fitness
        #                 eval_function = toolbox.evaluateMulti  # switch evaluation method
        #         else:
        #             max_times = constants.MAX_FIT_TIMES
        #     else:
        #         if abs(fit_last-fit_best) >= (fit_last * constants.NOV_OFFSET):
        #             eval_function = toolbox.evaluate

        # SWITCH FUNCTION AVG
        fit_prev = fit_best
        # selects the fittest individual in pop (minimum for minimization)
        fit_best = tools.selBest(pop, k=1)[0].fitness.values[0]
        if g > 0:
            if eval_function == toolbox.evaluate:
                if 0.95 <= fit_best / fit_prev <= 1.05:  # if they are similar
                    max_times -= 1
                    # switch to multi with novelty
                    if max_times == 0:
                        max_times = constants.MAX_FIT_TIMES
                        avg_last = sum([ind.fitness.values[0] for ind in pop])/constants.POP_SIZE
                        stdev_last = statistics.stdev([ind.fitness.values[0] for ind in pop])
                        eval_function = toolbox.evaluateMulti  # switch evaluation method
                else:
                    max_times = constants.MAX_FIT_TIMES
            else:
                avg_best = sum([ind.fitness.values[0] for ind in pop])/constants.POP_SIZE
                if abs(avg_last - avg_best) >= stdev_last/2:
                    eval_function = toolbox.evaluate

        # archive assessment
        if eval_function == toolbox.evaluateMulti:
            novelty_search.archive_assessment_best_in_pop(elite, archive)

        # NEW POP
        pop[:] = elite + offspring
        # delete archive duplicates entries
        # archive = list(set(archive))

        # SAVE STATISTICS
        # res = [ind.fitness.values for ind in pop]
        tot_f = [ind.fitness.values[0] for ind in pop]
        tot_n = [ind.fitness.values[1] for ind in pop]
        fits.append(sum(tot_f) / constants.POP_SIZE)
        mins.append(min(tot_f))
        maxs.append(max(tot_f))
        novs.append(sum(tot_n) / constants.POP_SIZE)
        arch_s.append(len(archive))

        # save stats
        # in case use copy.deepcopy()
        stats[g]["pop"] = pop[:]
        # stats[g]["fitness"] = res[:]

    # end ga

    ###############################################################
    #                   OUT, PLOTS and GRAPHS
    ###############################################################
    stats["final_archive"] = archive[:]
    stats["time"] = (datetime.now() - start_time).total_seconds()

    pop_plot = {"fits": [], "novs": []}
    best_plot = {"fits": [], "novs": []}
    bb_stats = dict()

    for pb in pop:
        pop_plot["fits"].append(pb.fitness.values[0])
        pop_plot["novs"].append(pb.fitness.values[1])

    bests = toolbox.select(pop, k=10)
    for i, bb in enumerate(bests):
        bb_stats[i] = dict()
        bb_stats[i]["individual"] = bb
        bb_stats[i]["fit"] = bb.fitness.values
        best_plot["fits"].append(bb.fitness.values[0])
        best_plot["novs"].append(bb.fitness.values[1])
        bb_stats[i]["seqs"] = markov.generate_with_weights(
            tps=tps, weights=bb, n_seq=constants.NUM_SEQS, occ_per_seq=gen_sequence_length, start_pool=start_pool)

    print("execution ended. output dir:", dir_out, "time elapsed:", stats["time"], "sec.")

    # save generated sequences
    with open(dir_out + "selected.json", "w") as fp:
        json.dump(bb_stats, fp, default=markov.serialize_sets)

    # result for weights progress
    pop2 = dict()
    for kk in range(constants.NGEN)[0::10]:
        pop2[kk] = stats[kk]["pop"]
    with open(dir_out + "pop.json", "w") as fp:
        json.dump(pop2, fp, default=markov.serialize_sets)
    # save stats
    with open(dir_out + "stats.json", "w") as fp:
        json.dump(stats, fp, default=markov.serialize_sets)

    # plots.plot_fits(dir_out, constants.NGEN, fits, novs, novelty_method)
    plots.plot_data(dir_out, constants.NGEN, fits, novs, arch_s, novelty_method)
    plots.plot_data2(dir_out, constants.NGEN, fits,mins,maxs, novs, arch_s)
    plots.plot_pareto(dir_out, pop_plot, best_plot, novelty_method)


if __name__ == "__main__":
    # run_ga("input", 43, "multi_log_switch")
    run_ga("all_irish-notes_and_durations-abc", 11, "multi_log_switch")
