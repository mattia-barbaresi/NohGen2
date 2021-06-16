# count pre- and post- occurrences for each word to find form classes
import numpy as np
import pprint
import metrics
import utils

pp = pprint.PrettyPrinter(indent=2)


# records pre- and post- lists of words (and the number of occurrences of each of them) that come
# before and after each word in sequences
def distributional_context(sequences, order=1):
    res = dict()
    for seq in sequences:
        for el in seq:
            if el not in res:
                res[el] = dict()
                res[el]["sx"] = dict()
                res[el]["dx"] = dict()
                # for all other strings
                for search_seq in sequences:
                    values = np.array(search_seq)
                    # for each found match
                    for index in np.where(values == el)[0]:
                        # count all pre e post order-context words
                        for i in range(1, order + 1):
                            # sx occurrence
                            if index >= i:
                                if values[index - i] in res[el]["sx"]:
                                    res[el]["sx"][values[index - i]] += 1
                                else:
                                    res[el]["sx"][values[index - i]] = 1
                            # dx occurrence
                            if index < len(values) - i:
                                if values[index + i] in res[el]["dx"]:
                                    res[el]["dx"][values[index + i]] += 1
                                else:
                                    res[el]["dx"][values[index + i]] = 1
    return res


# evaluates form classes
# returns words with no sx context
def start_words_old(dist_ctx):
    # print initial and ending classes
    res = set()
    for word in dist_ctx.items():
        if not word[1]["sx"]:
            res.add(word[0])
    return res


def start_words(classes, patterns):
    # print initial and ending classes
    res = set()
    # return members of starting classes in patterns
    pti = [x.strip(" ").split(" ")[0] for x in patterns]
    for i in pti:
        res.update(classes[int(i)])
    return res


def search(k,arr):
    for s in arr.items():
        if k in s[1]:
            return True
    return False


def form_classes(dist_ctx, fc_thr):
    angles = dict()
    # selects words and uses them as a coordinate vector
    # limit the number of coords
    coords = list(dist_ctx.keys())
    # evaluates words distance, context similarity
    for itm1 in dist_ctx.items():
        if itm1[0] not in angles:
            angles[itm1[0]] = dict()
            for itm2 in dist_ctx.items():
                if (itm2[0] != itm1[0]) and (itm2[0] not in angles[itm1[0]]):
                    # calculates pre- and post- contexts similarity (angles)
                    v1 = utils.angle_from_dict(itm1[1]["sx"],itm2[1]["sx"], coords)
                    v2 = utils.angle_from_dict(itm1[1]["dx"],itm2[1]["dx"], coords)
                    angles[itm1[0]][itm2[0]] = (v1 + v2)/2
    # evaluates form classes
    res = dict()
    idx = 1
    for k,values in angles.items():
        if not search(k,res):
            sim = set()
            sim.add(k)
            for x in values.items():
                if float(x[1]) <= fc_thr:
                    sim.add(x[0])
            res[idx] = list(sim)
            idx += 1
    # print("res: ")
    # pp.pprint(res)
    return res


def classes_index(classes, word):
    for cl in classes.items():
        if word in cl[1]:
            return cl[0]
    return -1


def classes_patterns(sequences, classes):
    res = set()
    for seq in sequences:
        pattern = ""
        for el in seq:
            val = classes_index(classes, el)
            if val != -1:
                pattern += " " + str(val)
            else:
                print("ERROR")
        pattern = pattern.strip(" ")
        res.add(pattern)
    # print("class patterns: ")
    # pp.pprint(res)
    return list(res)


# evaluate generated sequences with form classes and pattern
# return 1 if the generated pattern is in patterns set
# def evaluate_sequences(sequences, classes, patterns):
#     res = []
#     for seq in sequences:
#         iseq = seq
#         res_patt = ""
#         while len(iseq) > 0:
#             iseq2 = iseq
#             for cl in classes.items():
#                 fnd = False
#                 i = 0
#                 lst = list(cl[1])
#                 while i < len(lst) and (not fnd):
#                     if iseq.find(lst[i]) == 0:
#                         fnd = True
#                         res_patt += " " + str(cl[0])
#                         iseq = iseq[len(lst[i]):].strip(" ")
#                     else:
#                         i += 1
#             if iseq2 == iseq:
#                 return 0
#         res.append(1 if res_patt.strip(" ") in patterns else 0)
#     return res

# evaluate generated sequences with form classes and pattern using str_similarity
def evaluate_sequences2(sequences, classes, patterns):
    res = []
    for seq in sequences:
        iseq = seq
        res_patt = ""
        # translate sequence in a pattern
        while len(iseq) > 0:
            iseq2 = iseq
            fnd = False
            for cl in classes.items():
                i = 0
                lst = list(cl[1])
                while i < len(lst) and (not fnd):
                    if iseq.find(lst[i]) == 0:
                        fnd = True
                        res_patt += " " + str(cl[0])
                        iseq = iseq[len(lst[i]):].strip(" ")
                    else:
                        i += 1
            if iseq2 == iseq:
                # rewrite symbol, no translation occurred
                sym = iseq.strip(" ").split(" ", 1)
                res_patt += " " + sym[0]
                if len(sym) > 1:
                    iseq = sym[1]
                else:
                    iseq = ""
        # compute similarity
        vals = []
        for x in patterns:
            vals.append(metrics.str_similarity(res_patt.strip(" "), x))
        res.append(max(vals))
    return res


