"""
Runs multiple GAs on params values
"""
import fnmatch
import os
from datetime import datetime
import generate_models
from main import run_ga
import multiprocessing as mp


def _apply_fun(x):
    # fname, ranseed, novmeth
    run_ga(x[0], x[1], x[2])


def main():
    data = []
    pool = mp.Pool(mp.cpu_count()-1)
    start_time = datetime.now()

    seeds = [7, 43, 3465]
    methods = ["multi_log_switch"]
    files = ["irish", "bicinia", "all_irish-notes_and_durations-abc", "all_songs_in_G"]

    for fl in files:
        for rs in seeds:
            for nov_method in methods:
                data.append([fl, rs, nov_method])

    # multiprocessing
    pool.map(_apply_fun, data)
    pool.close()
    pool.join()
    print("batch time elapsed :", (datetime.now() - start_time).total_seconds(), "sec.")


if __name__ == "__main__":
    main()
