import time
from datetime import datetime
import json
import pprint
import markov
import os
import complexity
from shutil import copyfile
import utils
# import networkx as nx

# init, input, output
pp = pprint.PrettyPrinter(indent=2)
file_in = "input/input.txt"
sep = ""
dir_out = "results_" + datetime.now().strftime("%Y%m%d-%H.%M.%S") + "/"
os.mkdir(dir_out)
copyfile(file_in, dir_out + "input.txt")
# read
sequences = utils.read_from_file(file_in, sep)
seqs_reversed = utils.read_from_file(file_in, sep, reverse=True)
start_time = time.time()


##################################################################
# for epsilon_machine analysis
alf = set()
for s in sequences:
    for si in s:
        alf.add(si)

with open(dir_out + "alf.txt", "w") as fp:
    for i in alf:
        fp.write(i + " ")
##################################################################
# complexity
w = []
for sq in sequences:
    w.extend(sq)

results = dict()
results["entropy"] = complexity.entropy(w)
results["block_entropy_2"] = complexity.block_entropy(w, 2)
results["block_entropy_3"] = complexity.block_entropy(w, 3)
results["entropy_rate"] = complexity.entropy_rate(w)
results["predictive_information"] = complexity.predictive_information(w)
results["mutual_information(x,x+1)"] = complexity.mutual_information(w[:-1], w[1:])

with open(dir_out + "stats.json", "w") as fp:
    json.dump(results, fp, default=markov.serialize_sets, ensure_ascii=False)
##################################################################

# # chunk strength
# tf = markov.markov_chunk_strength(sequences)
# res = dict()
# res2 = []
# pp.pprint(tf)
# print("---")
# for it in tf.items():
#     if it[0] == 0:
#         res[0] = sorted(it[1].items(), key=lambda item: float(item[1]), reverse=True)
#     else:
#         res[it[0]] = dict()
#         for obj in it[1].items():
#             res[it[0]][obj[0]] = sorted(obj[1].items(), key=lambda item: float(item[1]), reverse=True)
#         # res2.res[it[0]]
# pp.pprint(res)


# compute tokens
tkn_tf, tkn_tf_seq, tokens, token_vocabulary, tokenized = markov.compute(sequences, dir_out, "tokens")
tkn_tf_r, tkn_tf_seq_r, tokens_r, tkn_vocabulary_r, tokenized_r = markov.compute(seqs_reversed, dir_out, "tokens_rev")
# convert tokenized to arr
arr = utils.dict_to_arr(tokenized)
print("--- tokenized")
pp.pprint(arr)
print("-----------------------")

# compute patterns
pat_tf, pat_tf_seq, patterns, pattern_vocabulary, patternized = markov.compute(arr, dir_out, "patterns")
tf = pat_tf
vocabulary = token_vocabulary

# generate new sequences
generated = markov.generate(tf, 30, occ_per_seq=10)
# translate to tokens
t2 = markov.translate(generated, vocabulary)
with open(dir_out + "generated.json", "w") as fp:
    json.dump(generated, fp, default=markov.serialize_sets)
with open(dir_out + "translated.json", "w") as fp:
    json.dump(t2, fp, default=markov.serialize_sets, ensure_ascii=False)


# # compute and generate in loop
# loop_seqs = sequences
# for x in range(0,10):
#     _tf, _tf_seq, _tokens, _vocabulary, _tokenized = markov.compute(loop_seqs, write_to_file=False)
#     loop_seqs = utils.dict_to_arr(_tokenized)
#     _tf2, _tf_seq2, _tokens2, _vocabulary2, _tokenized2 = markov.compute(loop_seqs, write_to_file=False)
#     _gen = markov.generate(_tf2, 10, occ_per_seq=10)
#     loop_seqs = utils.generated_to_arr(markov.translate(_gen, _vocabulary))
#
# pp.pprint(loop_seqs)


# output  and to console
print("time elapsed :", time.time() - start_time)

# # for drawing graphs
# for ind,item in tkn_tf.items():
#     if ind > 0:
#         G = nx.DiGraph()
#         for it in item:
#             G.add_edges_from([(it[0],k[0]) for k in it])
#         nx.draw(G)
