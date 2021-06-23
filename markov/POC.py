import time
from datetime import datetime
import json
import pprint
import plots
from markov import mkv
import fc as fc
import os
from shutil import copyfile
import utils
import complexity
import matplotlib.pyplot as plt
# import networkx as nx


###############################################################
#               init, input, output
###############################################################
pp = pprint.PrettyPrinter(indent=2)
file_in = "../data/input.txt"
sep = ""
dir_out = "../data/out_POC/input" + datetime.now().strftime("%Y%m%d-%H.%M.%S") + "/"
os.mkdir(dir_out)
copyfile(file_in, dir_out + "input.txt")

# read
sequences = utils.read_from_file(file_in, sep)
start_time = time.time()


##################################################################
#                       compute tokens and patterns
##################################################################
tkn_tf, tkn_tf_seq, tks, tkn_voc, tknzd, tkn_cls, cls_patt = mkv.compute_poc(sequences, dir_out, "tok")
# convert tokenized to arr
# arr = utils.dict_to_arr(tknzd)

# compute patterns
# pat_tf, pat_tf_seq, patterns, pattern_vocabulary, ptnzd, ptt_cls, patt_cls_patt = mkv.compute_poc(tknzd[3], dir_out,"patt")

# compute sl on class patterns
# patt_arr = []
# for ps in cls_patt:
#     patt_arr.append(ps.split(" "))
# pat_tf, pat_tf_seq, patterns, pattern_vocabulary, ptnzd, ptt_cls, patt_cls_patt = mkv.compute_poc(patt_arr, dir_out,"classPatternSL")


##################################################################
#                           generate
##################################################################
tps = tkn_tf
start_p = tkn_cls["sp"]
voc = []
# generate new sequences
bw = [0.002, 0.001, 0.017, 0.12, 0.13, 0.73]
ww = [1, 0, 0, 0, 0, 0]


nsq = 1000
ops = len(sequences[1])

good_sequences = mkv.generate_with_weights(tps=tps, weights=bw, voc=voc, n_seq=nsq, occ_per_seq=ops, start_pool=start_p)
bad_sequences = mkv.generate_with_weights(tps=tps, weights=ww, voc=voc, n_seq=nsq, occ_per_seq=ops, start_pool=start_p)
with open(dir_out + "good_seqs.json", "w") as fp:
    json.dump(good_sequences, fp,)
with open(dir_out + "bad_seqs.json", "w") as fp:
    json.dump(bad_sequences, fp)

# print("good_seqs:", good_sequences)
# print("bad_seqs:", bad_sequences)
# # translate to tokens
# # t2 = mkv.translate({0:generated}, voc)
# # pp.pprint(t2)
# # print("translated:")
# pp.pprint(generated)

t0 = datetime.now()
eval_res = fc.evaluate_sequences(good_sequences, tkn_cls["fc"], cls_patt)
eval_res_w = fc.evaluate_sequences(bad_sequences, tkn_cls["fc"], cls_patt)
print("evaluate_sequences: ", datetime.now() - t0)
print("res:",sum(eval_res)/nsq)
print("resw:",sum(eval_res_w)/nsq)
print("-------------------------------------------------------------")

t1 = datetime.now()
eval_res = fc.evaluate_sequences2(good_sequences, tkn_cls["fc"], cls_patt)
eval_res_w = fc.evaluate_sequences2(bad_sequences, tkn_cls["fc"], cls_patt)
print("evaluate_sequences2: ", datetime.now() - t1)
print("res:",sum(eval_res)/nsq)
print("resw:",sum(eval_res_w)/nsq)
print("-------------------------------------------------------------")

t2 = datetime.now()
eval_res2 = mkv.sequences_markov_support_with_min_default(good_sequences, tps)
eval_res2_w = mkv.sequences_markov_support_with_min_default(bad_sequences, tps)
print("sequences_markov_support_with_min_default: ", datetime.now() - t2)
print("res2:",eval_res2)
print("res2w:",eval_res2_w)
print("-------------------------------------------------------------")

t5 = datetime.now()
eval_res5 = mkv.sequences_markov_support_log(good_sequences, tps)
eval_res5_w = mkv.sequences_markov_support_log(bad_sequences, tps)
print("sequences_markov_support_log: ", datetime.now() - t5)
print("res5:", eval_res5)
print("res5w:", eval_res5_w)
print("-------------------------------------------------------------")

t5e = datetime.now()
eval_res5 = mkv.sequences_markov_support_entropy(good_sequences, tps)
eval_res5_w = mkv.sequences_markov_support_entropy(bad_sequences, tps)
print("sequences_markov_support_entropy: ", datetime.now() - t5e)
print("res5:", eval_res5)
print("res5w:", eval_res5_w)
print("-------------------------------------------------------------")

weights = [1,1,1,1,1,1]
t3 = datetime.now()
eval_res3 = mkv.sequences_markov_support_with_switches(good_sequences, tps, weights)
eval_res3_w = mkv.sequences_markov_support_with_switches(bad_sequences, tps, weights)
print("sequences_markov_support_with_switches: ", datetime.now() - t3)
print("res3:",eval_res3)
print("res3w:",eval_res3_w)
print("-------------------------------------------------------------")


t4 = datetime.now()
eval_res4 = mkv.sequences_markov_support_per_order(good_sequences, tps, weights)
eval_res4_w = mkv.sequences_markov_support_per_order(bad_sequences, tps, weights)
print("sequences_markov_support_per_order: ", datetime.now() - t4)
print("eval_res4:")
pp.pprint(eval_res4)
print("eval_res4_w:")
pp.pprint(eval_res4_w)

#
# with open(dir_out + "generated.json", "w") as fp:
#     json.dump(generated, fp, default=mkv.serialize_sets)
# # with open(dir_out + "translated.json", "w") as fp:
# #     json.dump(t2, fp, default=markov.serialize_sets, ensure_ascii=False)
#
#
# plots.plot_tps(dir_out, tkn_tf_seq, file_name="tokens")
# plots.plot_tps(dir_out, pat_tf_seq, file_name="patterns")
##################################################################
#               compute and generate in loop
##################################################################
# loop_seqs = sequences
# for x in range(0,10):
#     _tf, _tf_seq, _tokens, _vocabulary, _tokenized = markov.compute(loop_seqs, write_to_file=False)
#     loop_seqs = utils.dict_to_arr(_tokenized)
#     _tf2, _tf_seq2, _tokens2, _vocabulary2, _tokenized2 = markov.compute(loop_seqs, write_to_file=False)
#     _gen = markov.generate(_tf2, 10, occ_per_seq=10)
#     loop_seqs = utils.generated_to_arr(markov.translate(_gen, _vocabulary))
#
# pp.pprint(loop_seqs)

###############################################################
#                   OUT, PLOTS and GRAPHS
###############################################################
# print("time elapsed :", time.time() - start_time)


# fig = plt.figure()
# ax = plt.axes()
# plt.grid(b=True)
# # plt.xticks(range(0, 20, 1))
# for itm in tkn_tf_seq[3]:
#     plt.plot(list(map(lambda x: x if x != "-" else 0, itm)))
# # # to add labels on graph
# # # https://queirozf.com/entries/add-labels-and-text-to-matplotlib-plots-annotation-examples
# plt.show()

# import networkx as nx
# # for drawing graphs
# for ind,item in tkn_tf.items():
#     if ind > 0:
#         G = nx.DiGraph()
#         for it in item:
#             G.add_edges_from([(it[0],k[0]) for k in it])
#         nx.draw(G)
