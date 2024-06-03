import pickle, os, sys
from fractions import Fraction as frac
import numpy as np
import re


if __name__ == '__main__':
    fname = sys.argv[1]
    with open(fname, "rb") as fp:
        class_detail = pickle.load(fp)
    print(class_detail)
