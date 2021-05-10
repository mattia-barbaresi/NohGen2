# guide: https://realpython.com/python-nltk-sentiment-analysis/

from pprint import pprint
import nltk

import utils

sequences = utils.read_from_file("input/input.txt", "")

words = nltk.word_tokenize(sequences)
fd = nltk.FreqDist(words)
