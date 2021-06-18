# used to collect result and stats over models
import fnmatch
import os
from pandas import DataFrame
from markov import mkv


def _compute(dir_name):
    tf, cls, pts, avgl = mkv.load_model(dir_name)
    ncls = [len(x[1]) for x in cls["fc"].items()]
    npats = len(pts)
    return {"file":dir_name,"nc":ncls, "np": npats}


stats = []
# collect stats
for fn in fnmatch.filter(os.listdir("."), "*_*_*"):
    stats.append(_compute(fn))

df = DataFrame(stats)

df.to_excel("stats.xlsx")
