from datetime import datetime
import os
import random
from shutil import copyfile

import constants
import plots
import utils
from Parameters import Parameters
from markov import mkv


def create(file_name, file_in_sep, random_seed, params):
    """Generate tps, classes and patterns from sequences in file_in"""

    # set random
    random.seed(random_seed)

    # Create target dir if don't exist
    dir_out = "data/models/" + file_name + "_" + str(random_seed) + "_" +\
              str(params.mkv_thr) + "_" + str(params.fc_thr) + "_" + str(params.fc_n_ctx) + "_" + str(params.fc_seg_ord) + "/"

    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    else:
        print("Directory ", dir_out, "already exists")

    # calculate model and form classes
    ti = datetime.now()
    sequences = utils.read_from_file("data/" + file_name + ".txt", separator=file_in_sep)
    tp, tps, cls, ptns = mkv.compute(sequences, params, dir_name=dir_out + "model/")
    print("Model of " + file_name + " computed... time: ", (datetime.now() - ti).total_seconds(), "s.")

    # plots.plot_tps(dir_out, tps)

    return dir_out


if __name__ == "__main__":
    pars = Parameters()
    create("bicinia", " ", 8, pars)
