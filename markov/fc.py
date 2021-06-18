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


def find_in_class(seq, classes):
    for cl in classes.items():
        i = 0
        lst = list(cl[1])
        while i < len(lst):
            if seq.find(lst[i]) == 0:
                return cl[0], lst[i]
            else:
                i += 1
    return -1, -1


def sequence2pattern(sequence, classes):
    pattern = ""
    iseq = sequence
    while len(iseq) > 0:
        cls, val = find_in_class(iseq, classes)
        if cls != -1:
            pattern += " " + str(cls)
            iseq = iseq[len(val):].strip(" ")
        else:
            return pattern.strip(" ")
    return pattern.strip(" ")


def sequence2pattern_with_replacement(sequence, classes):
    pattern = ""
    iseq = sequence
    while len(iseq) > 0:
        cls, val = find_in_class(iseq, classes)
        if cls != -1:
            pattern += " " + str(cls)
            iseq = iseq[len(val):].strip(" ")
        else:
            pattern += " \uFFFD"
            iseq = iseq[2:].strip(" ")
    return pattern.strip(" ")


# evaluate generated sequences with form classes and pattern
# return 1 if the generated pattern is in patterns set
def evaluate_sequences(sequences, classes, patterns):
    res = []
    for seq in sequences:
        res_patt = sequence2pattern(seq, classes)
        res.append(1 if res_patt in patterns else 0)
    return res


# evaluate generated sequences with form classes and pattern using str_similarity
def evaluate_sequences2(sequences, classes, patterns):
    res = []
    for seq in sequences:
        # translate sequence in a pattern
        res_patt = sequence2pattern_with_replacement(seq, classes)
        # compute similarity
        vals = []
        for x in patterns:
            vals.append(metrics.str_similarity(res_patt.strip(" "), x))
        res.append(max(vals))
    return res
