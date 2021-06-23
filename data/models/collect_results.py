# used to collect result and stats over models
import fnmatch
import os
from pandas import DataFrame
from markov import mkv
import matplotlib.pyplot as plt


def _compute(dir_name):
    tf, cls, pts, avgl = mkv.load_model(dir_name)
    ncls = [len(x[1]) for x in cls["fc"].items()]
    npats = len(pts)
    return {"file":dir_name,"nc":ncls, "np": npats}


def _class_counter(dir_name):
    tf, cls, pts, avgl = mkv.load_model(dir_name)
    dat = []
    ind = []
    plt.clf()
    for cl in cls['fc'].items():
        dat.append(len(cl[1]))
        ind.append(cl[0])
    plt.bar(ind,dat)
    plt.savefig(dir_name + "/cls_hist")


stats = []
# collect stats
for fn in fnmatch.filter(os.listdir("."), "all_irish-notes_and_durations-abc_*"):
    _class_counter(fn)
    # stats.append(_compute(fn))
#
# df = DataFrame(stats)
#
# df.to_excel("stats.xlsx")
