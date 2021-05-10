import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

import utils

sequences = utils.read_from_file("input/input.txt", "")

# DCE
d = 3
tau = 1

res = dict()

for i, seq in enumerate(sequences):
    res[i] = {"x": [], "y": [], "z": []}
    s = [ord(x) for x in seq]
    print(s)
    A = np.array(s)
    ll = A[(np.arange(d) * (tau + 1)) + np.arange(np.max(A.shape[0] - (d - 1) * (tau + 1), 0)).reshape(-1, 1)]
    print(ll)
    for p in ll:
        res[i]["x"].append(p[0])
        res[i]["y"].append(p[1])
        res[i]["z"].append(p[2])

# Make the plot
fig = plt.figure()
ax = fig.gca(projection='3d')
for itm in res.items():
    ax.plot(itm[1]["x"], itm[1]["y"], itm[1]["z"])

# Set the angle of the camera
# ax.view_init(30, 70)
plt.xticks(range(100, 120))
plt.yticks(range(100, 120))

plt.show()
