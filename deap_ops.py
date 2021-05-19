import constants
import numpy as np


# fun for creating individual
def dirichlet_individual():
    v = np.random.dirichlet(np.ones(constants.IND_SIZE), size=None)
    # for floating error
    v[-1] = v[-1] + (1.0 - sum(v))
    return v
