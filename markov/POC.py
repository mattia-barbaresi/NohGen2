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
dir_out = "../data/out_POC/input_" + datetime.now().strftime("%Y%m%d-%H.%M.%S") + "/"
os.mkdir(dir_out)
copyfile(file_in, dir_out + "input.txt")
# read
sequences = utils.read_from_file(file_in, sep)
# seqs_reversed = utils.read_from_file(file_in, sep, reverse=True)
start_time = time.time()

###############################################################
#               epsilon_machine analysis
###############################################################
# alf = set()
# for s in sequences:
#     for si in s:
#         alf.add(si)
# with open(dir_out + "alf.txt", "w") as fp:
#     for i in alf:
#         fp.write(i + " ")

###############################################################
#                       complexity
###############################################################
# w = []
# for sq in sequences:
#     w.extend(sq)
#
# results = dict()
# results["entropy"] = complexity.entropy(w)
# results["disequilibrium"] = complexity.disequilibrium(w)
# results["block_entropy_2"] = complexity.block_entropy(w, 2)
# results["block_entropy_3"] = complexity.block_entropy(w, 3)
# results["entropy_rate"] = complexity.entropy_rate(w)
# results["predictive_information"] = complexity.predictive_information(w)
# results["mutual_information(x,x+1)"] = complexity.mutual_information(w[:-1], w[1:])
#
# with open(dir_out + "stats.json", "w") as fp:
#     json.dump(results, fp, default=mkv.serialize_sets, ensure_ascii=False)

##################################################################
#                   chunk strength
##################################################################
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
# rs = mkv.compute_matlab_tps(sequences,3)
# print(rs)

##################################################################
#                       compute tokens and patterns
##################################################################
# tkn_tf,tf_seqs, tkn_cls, cls_patt = mkv.compute(sequences, dir_out)
tkn_tf, tkn_tf_seq, tks, tkn_voc, tknzd, tkn_cls, cls_patt = mkv.compute_poc(sequences, dir_out, "tok")
# convert tokenized to arr
# arr = utils.dict_to_arr(tknzd)
# compute patterns
pat_tf, pat_tf_seq, patterns, pattern_vocabulary, ptnzd, ptt_cls, patt_cls_patt = mkv.compute_poc(tknzd[3], dir_out,"patt")


##################################################################
#                           generate
##################################################################
tps = tkn_tf
start_p = tkn_cls["sp"]
voc = []
# generate new sequences
bw = [0, 0, 0, 0, 0, 1]
ww = [1, 0, 0, 0, 0, 0]


nsq = 1000

good_sequences = mkv.generate_with_weights(tps=tps, weights=bw, voc=voc, n_seq=nsq, occ_per_seq=15, start_pool=start_p)
bad_sequences = mkv.generate_with_weights(tps=tps, weights=ww, voc=voc, n_seq=nsq, occ_per_seq=15, start_pool=start_p)
print("good_seqs:", good_sequences)
print("bad_seqs:", bad_sequences)
# # translate to tokens
# # t2 = mkv.translate({0:generated}, voc)
# # pp.pprint(t2)
# # print("translated:")
# pp.pprint(generated)

# these below are the first sequences of input
# good_sequences = [
# "m e r h o x t i d l u m r u d",
# "m e r n e b r e l s o t f a l",
# "m e r l e v r e l s o t t a f",
# "k o f l e v r e l z o r f a l",
# "k o f h o x t i d l u m r u d",
# "d a z h o x j e s s o t t a f",
# "d a z l e v r e l s o t r u d",
# "d a z h o x r e l l u m r u d",
# "m e r l e v r e l z o r t a f",
# "d a z l e v r e l s o t t a f",
# ]
#
# bad_sequences = [
# "m e r t i d h o x r u d l u m",
# "m e r r e l n e b f a l s o t",
# "m e r r e l l e v t a f s o t",
# "k o f r e l l e v f a l z o r",
# "k o f t i d h o x r u d l u m",
# "d a z j e s h o x t a f s o t",
# "d a z r e l l e v r u d s o t",
# "d a z r e l h o x r u d l u m",
# "m e r r e l l e v t a f z o r",
# "d a z r e l l e v t a f s o t",
# ]


t1 = datetime.now()
eval_res = fc.evaluate_sequences2(good_sequences, tkn_cls["fc"], cls_patt)
eval_res_w = fc.evaluate_sequences2(bad_sequences, tkn_cls["fc"], cls_patt)
print("evaluate_sequences2: ", datetime.now() - t1)
print("res",sum(eval_res)/nsq)
print("resw",sum(eval_res_w)/nsq)
print("-------------------------------------------------------------")

t2 = datetime.now()
eval_res2 = mkv.sequences_markov_support_with_min_default(good_sequences, tps)
eval_res2_w = mkv.sequences_markov_support_with_min_default(bad_sequences, tps)
print("sequences_markov_support: ", datetime.now() - t2)
print("res2:",eval_res2)
print("res2w:",eval_res2_w)
print("-------------------------------------------------------------")

t5 = datetime.now()
eval_res5 = mkv.sequences_markov_support_log(good_sequences, tps)
eval_res5_w = mkv.sequences_markov_support_log(bad_sequences, tps)
print("sequences_markov_support5: ", datetime.now() - t5)
print("res5:", eval_res5)
print("res5w:", eval_res5_w)
print("-------------------------------------------------------------")

t3 = datetime.now()
eval_res3 = mkv.sequences_markov_support_with_switches(good_sequences, tps)
eval_res3_w = mkv.sequences_markov_support_with_switches(bad_sequences, tps)
print("sequences_markov_support2: ", datetime.now() - t3)
print("res3:",eval_res3)
print("res3w:",eval_res3_w)
print("-------------------------------------------------------------")

t4 = datetime.now()
eval_res4 = mkv.sequences_markov_support_per_order(good_sequences, tps)
eval_res4_w = mkv.sequences_markov_support_per_order(bad_sequences, tps)
print("sequences_markov_support3: ", datetime.now() - t4)
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
