import csv
import json
#
# fname = "data/out_old/results_20210528-19.44.11/model/tf.json"
# with open(fname) as f:
#     data = json.load(f)
#
#
# tps = open("sample.csv", "w")
# with open('tf.csv', 'w', newline='') as csvfile:
#     fn = data['0'].keys()
#     writer = csv.DictWriter(csvfile, fieldnames=fn)
#
#     writer.writeheader()
#     for d in data.items():
#         if d[0] == '0':
#             writer.writerow(d[1])
#         else:
#             for x in d[1].keys():
#                 writer.writerow(d[1][x])
import utils
from markov import mkv

sequences = utils.read_from_file("data/all_songs_in_G.txt", "")
# model
tps, tps_s, classes, patterns = mkv.compute(sequences,"data/out_POC/proofs/")
