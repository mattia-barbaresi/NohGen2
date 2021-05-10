import math
import numpy as np


# reads list of sequences from file
def read_from_file(file_name, separator=" ", reverse=False):
    """
    Read a list of strings and split into tokens using separator.

    ...

    Parameters
    ----------
    separator : str
        separator between tokens, used to split sequences
    file_name : str
        the name of the file to read
    reverse: bool
        reversed sequences
    """

    lst = []
    with open(file_name) as fp:
        for line in fp:
            if separator == "":
                a = list(line.strip())
            else:
                a = line.strip().split(separator)
            if a:
                if reverse:
                    a.reverse()
                lst.append(a)
    return lst


def dict_to_arr(d):
    arr = []
    for itm in d.items():
        for sq in itm[1]:
            if sq:
                arr.append(list(str(x) for x in sq))
    return arr


# convert generated into new list of sequences
def generated_to_arr(g):
    arr = []
    for itm in g.items():
        for sq in itm[1]:
            if len(sq) > 1:
                arr.append("".join(sq.split(" ")))
    return arr


# Function to find the magnitude
# of the given vector
def magnitude(arr):
    # Stores the final magnitude
    mag = 0

    # Traverse the array
    for i in range(len(arr)):
        mag += arr[i] * arr[i]

    # Return square root of mag
    return math.sqrt(mag)


# Function to find the dot
# product of two vectors


def dot_product(arr, brr):
    # Stores dot product
    product = 0
    size = min(len(arr), len(brr))
    # Traverse the array
    for i in range(size):
        product = product + (arr[i] * brr[i])

    # Return the product
    return product


def angle_from_vector(arr, brr):
    if len(arr) == len(brr) == 0:
        return 0  # empty arrays are equal

    # Stores dot product of two vectors
    dp = dot_product(arr, brr)

    # Stores magnitude of vector A
    magnitude_a = magnitude(arr)

    # Stores magnitude of vector B
    magnitude_b = magnitude(brr)

    # Stores angle between given vectors
    val = dp / (magnitude_a * magnitude_b)
    # round error fix
    if 1 - val <= 0.0001:
        val = 1
    angle = math.acos(val)

    # Print the angle
    # print('%.5f' % angle)
    return angle


def angle_from_dict(dict_a, dict_b):
    # if contexts have same values, calculate angle
    if np.array_equal(dict_a.keys(),dict_b.keys()):
        return angle_from_vector(list(dict_a.values()),list(dict_b.values()))
    else:
        return math.pi
