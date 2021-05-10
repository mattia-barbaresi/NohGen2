# count pre- and post- occurrences for each word to find form classes
from collections import OrderedDict

import numpy as np
import pprint

import utils

pp = pprint.PrettyPrinter(indent=2)


# records pre- and post- lists of words (and the number of occurrences of each of them) that come
# before and after each word in sequences
def distributional_context(sequences, order=1):
    res = OrderedDict()
    for seq in sequences:
        for el in seq:
            if el not in res:
                res[el] = dict()  # for preserving sequence order
                for i in range(1, order + 1):
                    res[el][-i] = dict()
                    res[el][+i] = dict()
                    for search_seq in sequences:
                        values = np.array(search_seq)
                        for index in np.where(values == el)[0]:
                            # sx occurrence
                            if index >= i:
                                if values[index - i] in res[el][-i]:
                                    res[el][-i][values[index - i]] += 1
                                else:
                                    res[el][-i][values[index - i]] = 1
                            # dx occurrence
                            if index < len(values) - i:
                                if values[index + i] in res[el][+i]:
                                    res[el][+i][values[index + i]] += 1
                                else:
                                    res[el][+i][values[index + i]] = 1
                    # order results
                    res[el][+i] = OrderedDict(sorted(res[el][+i].items(), key=lambda x: x[0], reverse=True))
                    res[el][-i] = OrderedDict(sorted(res[el][-i].items(), key=lambda x: x[0], reverse=True))
    return res


# evaluates form classes
def first_last(dist_ctx):
    # print initial and ending classes
    first_set = []
    last_set = []
    pp.pprint(dist_ctx)
    for word in dist_ctx.items():
        if not word[1][-1]:
            first_set.append(word[0])
        if not word[1][1]:
            last_set.append(word[0])
    print("first: ", first_set)
    print("last: ", last_set)


def search(k,arr):
    for s in arr:
        if k in s:
            return True
    return False


def form_classes(dist_ctx):
    angles = dict()
    for itm1 in dist_ctx.items():
        if itm1[0] not in angles:
            angles[itm1[0]] = dict()
            for itm2 in dist_ctx.items():
                if (itm2[0] != itm1[0]) and (itm2[0] not in angles[itm1[0]]):
                    v1 = utils.angle_from_dict(itm1[1][-1], itm2[1][-1])
                    v2 = utils.angle_from_dict(itm1[1][+1], itm2[1][+1])
                    angles[itm1[0]][itm2[0]] = (v1 + v2)/2
    # print("angles: ")
    # pp.pprint(angles)
    res = []
    for k,values in angles.items():
        # print (values)
        if not search(k,res):
            sim = set()
            sim.add(k)
            sim.update(x[0] for x in values.items() if float(x[1]) < 2)
            res.append(sim)

    print("res: ")
    pp.pprint(res)
    return res

