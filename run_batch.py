"""
Runs multiple GAs per params values
"""
from datetime import datetime

import utils
from Params import Params
from main import run_ga
import multiprocessing as mp

from markov import mkv


def main():
    data = []
    pool = mp.Pool(mp.cpu_count())
    start_time = datetime.now()

    seeds = [7, 13]
    methods = ["genotype", "phenotype"]
    files = [
        # {"name": "input", "sep": ""},
        # {"name": "input2", "sep": ""},
        {"name": "all_songs_in_G", "sep": ""},
        # {"name": "irish", "sep": " "}
    ]

    # file name and separator
    for fl in files:
        # seed for random
        for rs in seeds:
            # novelty method
            for nov_method in methods:
                # run_ga(Params(file_in=fl, random_seed=rs, novelty_method=nov_method))
                data.append(Params(file_in=fl, random_seed=rs, novelty_method=nov_method))

    # multiprocessing
    pool.map(run_ga, data)

    pool.close()
    pool.join()
    print("time elapsed :", (datetime.now() - start_time).total_seconds(), "sec.")


if __name__ == "__main__":
    main()
