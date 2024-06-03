import pickle
from fractions import Fraction as frac

# class_detail = {"num_classes": 3, "class_thresholds": [1718, 2757, float('inf')], "class_rates": [frac(941,1000),frac(7,125),frac(3,1000)], "clip_demand_factor": 0.01, "delta": 0.01}
# class_detail = {"num_classes": 2, "class_thresholds": [4790, float('inf')], "class_rates": [frac(23,25),frac(2,25)], "clip_demand_factor": 0.01, "delta": 0.01}
# class_detail = {"num_classes": 2, "class_thresholds": [3683,float('inf')], "class_rates": [frac(461,500),frac(39,500)], "clip_demand_factor": 0.01, "delta": 0.01}
# class_detail = {"num_classes": 2, "class_thresholds": [100,float('inf')], "class_rates": [frac(3,4),frac(1,4)], "clip_demand_factor": 0.01, "delta": 0.01}



class_detail = {"num_classes": 3, "class_thresholds": [1627,969473,float('inf')], "class_rates": [frac(19,20),frac(6,125),frac(1,500)], "clip_demand_factor": 0.5980, "delta": 0.23536}
file_name = "PCS_config_0e4a51_avg_jct_avg_pred_error_jct.pkl"
with open(file_name, "wb") as fp:
	pickle.dump(class_detail, fp)



class_detail = {"num_classes": 3, "class_thresholds": [2139,1533489,float('inf')], "class_rates": [frac(313,333),frac(19,333),frac(1,333)], "clip_demand_factor": 0.6326, "delta": 0.50166}
file_name = "PCS_config_0e4a51_avg_jct_avg_pred_error_bal.pkl"
with open(file_name, "wb") as fp:
	pickle.dump(class_detail, fp)



class_detail = {"num_classes": 3, "class_thresholds": [5758,4600151,float('inf')], "class_rates": [frac(19,20),frac(6,125),frac(1,500)], "clip_demand_factor": 0.3185, "delta": 0.16764}
file_name = "PCS_config_0e4a51_avg_jct_avg_pred_error_pred.pkl"
with open(file_name, "wb") as fp:
	pickle.dump(class_detail, fp)


# file_name = "{}_{}_{}.pkl".format(
# 	class_detail["num_classes"],
# 	",".join(list(map(str,class_detail["class_thresholds"]))),
# 	",".join(list(map(lambda r: str(r.numerator),class_detail["class_rates"]))))





