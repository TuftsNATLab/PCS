import pickle
import sys

with open(sys.argv[1], 'rb') as fp:
	class_detail = pickle.load(fp)


print(class_detail)