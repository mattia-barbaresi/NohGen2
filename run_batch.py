"""
Runs multiple GAs on params values
"""
import fnmatch
import os
from datetime import datetime
import generate_models
from Parameters import Parameters
from main import run_ga
import multiprocessing as mp


def _apply_fun(x):
    # fname, ranseed, novmeth
    run_ga(x[0],x[1],x[2])


def main():
    data = []
    pool = mp.Pool(mp.cpu_count())
    start_time = datetime.now()

    seeds = [7, 43]
    methods = ["genotype_jacc"]
    files = [
        # {"name": "input", "sep": ""},
        # {"name": "input2", "sep": ""},
        # {"name": "irish", "sep": " "},
        {"name": "bicinia", "sep": " "},
        {"name": "all_irish-notes_and_durations-abc", "sep": " "},

        # {"name": "all_songs_in_G", "sep": ""}, # generated only for seed = 7
    ]

    # file name and separator
    for fl in files:
        # seed for random
        for rs in seeds:

            # # generate input models
            # for mkv_thr in [0.75,0.85]:
            #     for fc_thr in [1.0, 1.2]:
            #         for fc_n_ctx in [3,4,5]:
            #             for fc_seg_lvl in [2,3,4]:
            #                 pars = Parameters(mkv_thr, fc_thr, fc_n_ctx, fc_seg_lvl)
            #                 generate_models.create(fl["name"], fl["sep"], rs, pars)

            # novelty method for each model
            # read models/README
            for fn in fnmatch.filter(os.listdir("data/models/"), fl["name"] + "_" + str(rs) + "_*"):
                for nov_method in methods:
                    data.append([fn, rs, nov_method])
    #
    # # multiprocessing
    pool.map(_apply_fun, data)
    pool.close()
    pool.join()
    print("batch time elapsed :", (datetime.now() - start_time).total_seconds(), "sec.")


if __name__ == "__main__":
    main()
