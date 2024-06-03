import numpy as np
import matplotlib.pyplot as plt
import os, sys, argparse
import operator

# app_id,num_samples,submit_time,dispatch_time,start_time,end_time,estimated_start_time,estimated_end_time,theoretical_duration,estimated_duration,actual_duration


def file_data_filter(fdata, col_idx, comparison_op, compare_value, delimiter=','):
    
    filtered_fdata = list()

    for entry in fdata:
        if comparison_op(float(entry.rstrip().split(delimiter)[col_idx]), compare_value):
            filtered_fdata.append(entry)
    return filtered_fdata


def parse_file_data(fname, delimiter=','):

    result_dictionary = {}

    with open(fname) as fd:
        keys, *file_data = fd.readlines()

    # file_data = file_data_filter(file_data, 1, operator.eq, 6)




    keys = keys.rstrip().split(delimiter)




    for i, key in enumerate(keys):

        if key == "status":
            continue

        result_dictionary[key] = list(map(lambda e: float(e.rstrip().split(delimiter)[i]), file_data))

    return result_dictionary

def cdf_queuing_delay(result_dictionary):
    queueing_delay = np.subtract(result_dictionary["start_time"], result_dictionary["submit_time"])
    queueing_delay.sort()
    cdf = np.linspace(1.0/len(queueing_delay),1.0,len(queueing_delay))
    return queueing_delay, cdf



def cdf_ACT(result_dictionary):
    TCT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    TCT.sort()
    cdf = np.linspace(1.0/len(TCT),1.0,len(TCT))
    return TCT, cdf



def cdf_queueing_delay_error(result_dictionary):

    actual_queueing_delay = np.subtract(result_dictionary["start_time"], result_dictionary["submit_time"])
    estimated_queueing_delay = np.subtract(result_dictionary["estimated_start_time"], result_dictionary["submit_time"])


    queueing_delay_error = list()
    for i in range(len(actual_queueing_delay)):

        queueing_delay_error.append((estimated_queueing_delay[i] - actual_queueing_delay[i]))

    queueing_delay_error.sort()
    cdf = np.linspace(1.0/len(queueing_delay_error),1.0,len(queueing_delay_error))
    return queueing_delay_error, cdf


def cdf_ml_app_time_error(result_dictionary):

    ml_app_time = result_dictionary["actual_duration"]
    estimated_ml_app_time = result_dictionary["estimated_duration"]

    error = np.subtract(estimated_ml_app_time, ml_app_time)
    error.sort()
    cdf = np.linspace(1.0/len(error),1.0,len(error))
    return error, cdf

def cdf_ml_app_time(result_dictionary):
    ml_app_time = np.subtract(result_dictionary["end_time"], result_dictionary["start_time"])
    ml_app_time.sort()
    cdf = np.linspace(1.0/len(ml_app_time),1.0,len(ml_app_time))
    return ml_app_time, cdf



def cdf_ACT_error(result_dictionary):
    actual_ACT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    estimated_ACT = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["submit_time"])
    ACT_error = [100.0 * abs(estimated_ACT[i] - actual_ACT[i]) / estimated_ACT[i] for i in range(len(estimated_ACT))]
    ACT_error.sort()

    cdf = np.linspace(1.0/len(ACT_error),1.0,len(ACT_error))
    return ACT_error, cdf


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('fnames', nargs="+", help = "input filenames", type=str)
    args = parser.parse_args()

    fnames = args.fnames

    print(fnames)

    colors = ['b','r','g','k','y','c','darkviolet']
    

    results = {}
    for fname in fnames:
        results[fname] = parse_file_data(fname)


    for func in [cdf_ACT]:
        for i, fname in enumerate(fnames):
            result_dictionary = results[fname]

            x, y = func(result_dictionary)
            label = fname.split('/')[-1].split('.')[0]

            plt.plot(x, y, color=colors[i], marker='o', markevery=1, label=label)

        plt.legend(loc="best")
        plt.ylim(0, 1.0)
        plt.xlabel("Time")
        plt.ylabel('CDF')
        plt.grid(alpha=.3, linestyle='--')
        plt.savefig('%s.png' % (func.__name__), dpi = 300)
        plt.figure()