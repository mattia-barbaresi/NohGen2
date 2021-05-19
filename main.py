from deap import base, creator, tools
import deap_ops

# fitness functions
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
# individual class
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("dirInd", deap_ops.dirichlet_individual)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.dirInd)

