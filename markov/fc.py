# count pre- and post- occurrences for each word to find form classes
import numpy as np
import pprint
import utils

pp = pprint.PrettyPrinter(indent=2)

# threshold for classes
THS = 1.1


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
                for i in range(1, order + 1):
                    for search_seq in sequences:
                        values = np.array(search_seq)
                        for index in np.where(values == el)[0]:
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
def start_words(dist_ctx):
    # print initial and ending classes
    res = set()
    for word in dist_ctx.items():
        if not word[1]["sx"]:
            res.add(word[0])
    return res


def search(k,arr):
    for s in arr.items():
        if k in s[1]:
            return True
    return False


def form_classes(dist_ctx):
    angles = dict()
    coords = dist_ctx.keys()
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
            sim.update(x[0] for x in values.items() if float(x[1]) < THS)
            res[idx] = sim
            idx += 1
    print("res: ")
    pp.pprint(res)
    return res


def classes_index(classes, word):
    for cl in classes.items():
        if word in cl[1]:
            return cl[0]
    return -1


def classes_patterns(classes,sequences):
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
    print("class patterns: ")
    pp.pprint(res)
    return res


# given a sequence calculate its pattern based on classes
def evaluate_seq(sequence, classes, patterns):
    iseq = sequence
    res_patt = ""
    while len(iseq) > 0:
        iseq2 = iseq
        for cl in classes.items():
            fnd = False
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
            return 0
    return 1 if res_patt.strip(" ") in patterns else 0


# # evaluate generated sequences with form classes and pattern
def evaluate_sequences(sequences, classes, patterns):
    res = []
    for seq in sequences:
        res.append(evaluate_seq(seq, classes, patterns))
    return res


# returns form classes and start pool
def cf_model(dist_ctx):
    return {"fc": form_classes(dist_ctx), "sp": start_words(dist_ctx)}
