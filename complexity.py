#!/usr/bin/env python

import bz2
from math import log


####################################################################################################################
def frequency(data):
    # Function computing the relative frequency of symbols of an array of data.
    #
    # NOTE: each element of the input array is taken as a string and as such used to compile the dictionary

    """Function computing the relative frequency of symbols of an array of data.
    NOTE: each element of the input array is taken as a string and as such used to compile the dictionary"""

    mylen = len(data)
    sdict = {}

    for x in data:
        x = str(x)
        if x in sdict:
            sdict[x] = sdict[x] + 1
        else:
            sdict[x] = 1

    freq_array = []
    for t in sdict.items():
        freq_array.append(1.0 * t[1] / mylen)

    return freq_array


####################################################################################################################
def entropy(data):
    # Function computing the entropy of an array of data.
    #
    # NOTE: each element of the input array is taken as a string and as such used to compile the dictionary

    """Function computing the entropy of an array of data.
    NOTE: each element of the input array is taken as a string and as such used to compile the dictionary"""

    mylen = len(data)
    sdict = {}

    for x in data:
        x = str(x)
        if x in sdict:
            sdict[x] = sdict[x] + 1
        else:
            sdict[x] = 1

    freq_array = []
    for t in sdict.items():
        freq_array.append(1.0 * t[1] / mylen)
    n = len(freq_array)

    s = 0.0
    for p in freq_array:
        s = s + p * log(p, 2)

    return -s


####################################################################################################################
def disequilibrium(data):
    # Function computing the disequilibrium of an array of data.
    #
    # NOTE: each element of the input array is taken as a string and as such used to compile the dictionary

    """Function computing the disequilibrium of an array of data.
     NOTE: each element of the input array is taken as a string and as such used to compile the dictionary."""

    mylen = len(data)
    sdict = {}

    for x in data:
        x = str(x)
        if x in sdict:
            sdict[x] = sdict[x] + 1
        else:
            sdict[x] = 1

    freq_array = []
    for t in sdict.items():
        freq_array.append(1.0 * t[1] / mylen)
    n = len(freq_array)

    z = 0.0
    for p in freq_array:
        z = z + pow(p - 1.0 / n, 2)

    return z


####################################################################################################################
def lmc(data):
    """Function computing the LMC of an array of data."""

    ent = entropy(data)
    diseq = disequilibrium(data)

    return ent * diseq


####################################################################################################################
def group(v):
    w = ''
    for e in v:
        w = w + ' ' + str(e)
    return w.strip()


####################################################################################################################
def block_entropy(data, L):
    # Function computing the L-length block entropy of an array of data.

    """Function computing the L-length block entropy of an array of data."""

    mylen = len(data)
    sdict = {}

    for i in range(0, mylen - L + 1):
        s = data[i:i + L]
        x = group(s)
        if x in sdict:
            sdict[x] = sdict[x] + 1
        else:
            sdict[x] = 1

    # print sdict

    #  print str(L) ##DEBUG##

    freq_array = []
    for t in sdict.items():
        freq_array.append(1.0 * t[1] / (mylen - L + 1))
        # print str(t[0]) + ':' + str(t[1])

    s = 0.0
    for p in freq_array:
        s = s + p * log(p, 2)

    return -s


####################################################################################################################
def entropy_rate(data, Llim=0):
    """Function estimating the entropy rate. As L cannot grow to infinity, we compute the value for L = Llim."""
    if Llim == 0:
        Llim = len(data)
    return block_entropy(data, Llim) / Llim


####################################################################################################################
def ngram_entropy(data, L):
    """Function computing the L-excess entropy hn = Hn+1 - Hn."""
    return (block_entropy(data, L + 1) - block_entropy(data, L))
    # return (block_entropy(data,L) - block_entropy(data,L-1))


###################################################################################################################
def hn(data, L):
    """Function computing h_n according to Lindgren as H(n) - H(n-1)."""
    if L < 2:
        return 0
    else:
        return (block_entropy(data, L) - block_entropy(data, L - 1))


###################################################################################################################
def correlation_information(data, L):
    """Function computing Kn = -delta^2 Sn, where Sn is the n-block entropy"""
    if L < 3:
        return 0
    else:
        return -block_entropy(data, L) + 2 * block_entropy(data, L - 1) - block_entropy(data, L - 2)


###################################################################################################################
def excess_entropy(data, Llim=0):
    # The function is computed according to the simple definition in terms of sum of differences of block entropies

    """Function estimating the excess entropy."""
    if Llim == 0:
        Llim = len(data)

    excent = 0
    for L in range(1, Llim):
        excent = excent + delta_block_entropy(data, L)

    return excent

    # er = entropy_rate(data,Llim)
    # entsum = 0
    # for L in range(1,Llim+1):
    # entsum = entsum + entropy_rate(data,L)
    # return entsum - Llim * er

    # return (block_entropy(data,Llim) - Llim * entropy_rate(data,Llim))


###################################################################################################################
def emc(data, Llim=0):
    """Function computing the EMC (see paper by Grassberger).
    As L cannot grow to infinity, we compute the value for L = Llim. Default Llim = len(data)/4."""
    if Llim == 0:
        Llim = len(data)
    emc = -entropy(data)
    for L in range(1, Llim):
        emc = emc + delta_block_entropy(data, L)

    return emc


###################################################################################################################
def mutual_information(X, Y):
    """Computes the mutual information between X and Y, taken as lists of values.
    The array should be of the same length; if not the minimum is taken."""
    hx = entropy(X)
    hy = entropy(Y)
    xy = []
    l = min(len(X), len(Y))
    for i in range(0, l):
        xy.append(str(X[i]) + str(Y[i]))
        hxy = entropy(xy)
    return hx + hy - hxy


###################################################################################################################
def cross_correlation_xy(x, y, lag=1):
    s = 0
    for i in range(len(y) - lag):
        if x[i] == y[i + lag]:
            s = s + 1
        s = s / (1.0 * len(y))  # NOTE: we average
    return s


###################################################################################################################
def lcs(data, L):
    """Computes the number of distinct subsequences of length L in data."""

    n = len(data)
    sdict = {}
    for i in range(0, n - L + 1):
        s = data[i:i + L]
        x = group(s)
        if x in sdict:
            sdict[x] = sdict[x] + 1
        else:
            sdict[x] = 1

    # print sdict
    # print len(sdict)

    return 1.0 * len(sdict) / n


###################################################################################################################
def lc(data, L):
    """Computes the linguistic sequence complexity at length L of data."""

    lc_ret = 0
    for k in range(1, L + 1):
        lc_ret = lc_ret + lcs(data, k)

    return lc_ret


###################################################################################################################
def compression_factor(data):
    """Computes the compression factor of the data taken as string."""
    mystring = ''
    for x in data:
        mystring = mystring + str(x)
    size0 = float(len(mystring))
    size1 = float(len(bz2.compress(mystring)))

    return str(size0 / size1)


###################################################################################################################
def lz_complexity(data):
    """Computes the LZ76 complexity"""
    s = []
    for x in data:
        s.append(str(x))

    i, k, l = 0, 1, 1
    k_max = 1
    n = len(s) - 1
    c = 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k = k + 1
            if l + k >= n - 1:
                c = c + 1
                break
        else:
            if k > k_max:
                k_max = k
            i = i + 1
            if i == l:
                c = c + 1
                l = l + k_max
                if l + 1 > n:
                    break
                else:
                    i = 0
                    k = 1
                    k_max = 1
            else:
                k = 1
    return c


###################################################################################################################
def predictive_information(data):
    # """Computes the PI, as mutual information between X(t+1) and X(t), of the data taken as string."""

    l = len(data)
    x_t0 = data[0:l - 1]
    x_t1 = data[1:l]

    predictive_information = mutual_information(x_t1, x_t0)

    return predictive_information

###################################################################################################################
# def neural_complexity(traj):
# """Computes an approximation of the neural complexity considering only the partitions {1,n-1}.
# Trajectory is a list of strings which will be treated as char sequencences."""

# for x in data:
