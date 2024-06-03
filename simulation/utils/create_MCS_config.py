import pickle, os, sys
from fractions import Fraction as frac
import numpy as np
import re
# class_detail = {"num_classes": 3, "class_thresholds": [1718, 2757, float('inf')], "class_rates": [frac(941,1000),frac(7,125),frac(3,1000)]}

#THEMIS1 low to high perf
mcs_perf = "MCS_3_1718,2757,inf_941,7,3"
class_detail = {"num_classes": 3, "class_thresholds": [1718, 2757, float('inf')], "class_rates": [frac(941,1000),frac(7,125),frac(3,1000)]}
class_detail = {"num_classes": 2, "class_thresholds": [3122975, float('inf')], "class_rates": [frac(381, 1000), frac(619, 1000)]}


class_detail = {"num_classes": 5, "class_thresholds": [1286, 20231, 393176, 4388656, float('inf')], "class_rates": [frac(3, 250), frac(33, 1000), frac(11, 125), frac(59, 250), frac(631, 1000)]}

# mcs_est = "MCS_2_4790,inf_23,2"
# mcs_middle = "MCS_2_3683,inf_461,39"


#GAVEL cont single
# class_detail = {"num_classes": 2, "class_thresholds": [286, float('inf')], "class_rates": [frac(101,125),frac(24,125)]}
# class_detail = {"num_classes": 2, "class_thresholds": [529, float('inf')], "class_rates": [frac(177,200), frac(23,200)]}
# class_detail = {"num_classes": 3, "class_thresholds": [117, 5138, float('inf')], "class_rates": [frac(73, 77), frac(7, 143), frac(3, 1001)]}


num_classes = int(input("Enter number of classes (n>=1): "))
class_thresholds = list()

t = input("Enter thresholds (comma+space seperated): ")
class_thresholds = list(map(float, t.split(', ')))
class_thresholds = list(map(int, class_thresholds[:-1])) + [class_thresholds[-1]]

r = input("Enter rates (Fraction(n, d) comma+space seperated): ")


class_rates = list()
p = re.compile(r"Fraction(\d, \d)")
p = re.compile(r"\d+")
res = np.asarray(list(map(int, p.findall(r))))
res = np.reshape(res, (len(res)//2, 2))

for r in res:
	class_rates.append(frac(r[0], r[1]))


clip_demand_factor = None
clip_demand_factor = float(input("Enter clip demand factor ([1,0]): "))
delta = float(input("Enter delta ((0,1]): "))

print(num_classes)
print(class_thresholds)
print(class_rates)
print(sum(class_rates))
print(clip_demand_factor)
print(delta)

assert(sum(class_rates) == 1)

class_detail = {"num_classes": num_classes,
				"class_thresholds": class_thresholds,
				"class_rates": class_rates,
				"clip_demand_factor": clip_demand_factor,
				"delta": delta}


file_name = "{}_{}_{}_{}_{}.pkl".format(
    class_detail["num_classes"],
    ",".join(list(map(str,class_detail["class_thresholds"]))),
    ",".join(list(map(lambda r: str(r.numerator),class_detail["class_rates"]))),
    clip_demand_factor,
    delta)


print(file_name)

with open(file_name, "wb") as fp:
    pickle.dump(class_detail, fp)
