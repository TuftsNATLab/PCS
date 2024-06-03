import pickle
import os, sys


fname = sys.argv[1]

with open(fname, 'rb') as fp:
    config = pickle.load(fp)

print(config)

