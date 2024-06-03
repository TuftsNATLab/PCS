from unittest import result
import numpy as np
import matplotlib.pyplot as plt
import os, sys, argparse
import operator
import seaborn as sns

# app_id,num_samples,submit_time,dispatch_time,start_time,end_time,estimated_start_time,estimated_end_time,theoretical_duration,estimated_duration,actual_duration


# policies = ["MCS_avg_perf", "MCS_avg_est", "MCS_tail_perf", "MCS_tail_est", "FIFO", "SRSF", "AFS", "FS", "LAS", "THEMIS", "OPTIMUS"]

policy_tags = ["MCS_perf", "MCS_est", "MCS_middle", "MCS_tail_est", "MCS_other", "FIFO", "SRSF", "AFS", "FS", "LAS", "THEMIS", "OPTIMUS"]



labels = ["Fidelis-JCT",r"Fidelis-$\mathcal{E}_{pred}$", "Fidelis-bal","Fidelis-tail-pred", "Fidelis-other", "FIFO", "Tiresias", "AFS", "Max-Min", "LAS", "Themis", "Optimus"]

# palette_tab20 = sns.color_palette("tab20")
colors = sns.color_palette("tab20")
# colors = ["#3471eb","#7ea5f2",colors[2],colors[4],colors[6],colors[8],"black",colors[10]]
colors = ["cornflowerblue","royalblue","midnightblue","blue","#4d4848",colors[2],colors[4],colors[6],colors[8],"black",colors[10]]

linestyles=["solid","dotted","dashdot",(0,(3,5,1,5)),"solid","dotted","dashdot",(0,(3,5,1,5)),"solid","dotted","dashdot",(0,(3,5,1,5))]
markers=["o","o","o","o","s","s","s","s","D","D","D","D"]


# colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","black","tab:brown"]
# colors = ['lightcoral','peru','forestgreen','steelblue','plum','blue','mediumorchid','lightpink','black']





def get_index_policy(policy):

    for i, policy_tag in enumerate(policy_tags):
        if policy_tag == policy:
            return i
    return i
    


def chronus_error():
    error = [10,25,50,100,500]

    cdf = [21.28/100.0,
            38.3/100.0,
            28.72/100.0,
            9.75/100.0,
            2.13]

    return error, cdf



def set_canvas(ax, x_label=None, y_label=None, x_lim=None, y_lim=None, y_ticks=None, x_ticks=None, legend_loc='best', legend_ncol=1, showgrid=True, legendfsize=15, showlegend=False):
    
    ax.set_facecolor(("#c8cbcf"))


    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    if showlegend:
        ax.legend(loc=legend_loc, ncol=legend_ncol, prop={'family': 'monospace', 'size': legendfsize})
    
    if x_label:
        ax.set_xlabel(x_label, fontsize=20, family="monospace")
    if y_label:
        ax.set_ylabel(y_label, fontsize=20, family="monospace")



    if x_ticks:
        ax.set_xticks(x_ticks[0])
        ax.set_xticklabels(x_ticks[1])

    if y_ticks:
        ax.set_yticks(y_ticks[0])
        ax.set_yticklabels(y_ticks[1])

    if x_lim:
        ax.set_xlim(x_lim[0],x_lim[1])
    if y_lim:
        ax.set_ylim(y_lim[0],y_lim[1])


    if showgrid:
        ax.grid(alpha=.5, color="white", linestyle='-', zorder=0)



def label_from_policies(policy):

    for i, policy_tag in enumerate(policy_tags):
        if policy_tag == policy:


            # print(f"policy: {policy} policy_tag: {policy_tag} label: {labels[i]}")

            return labels[i]
    return "Undefined"

def color_from_policies(policy):

    for i, policy_tag in enumerate(policy_tags):
        if policy_tag == policy:
            return colors[i]
    return colors[0]
    


def markevery_calc(N, total_marks=10):
    markevery = N//min(total_marks, N)

    mark_idx = list(range(0, N, markevery))
    mark_idx = mark_idx + list(map(lambda p: int(p*N//100), [90, 95, 99, 99.9]))

    mark_idx = mark_idx + [N-1]
    return mark_idx


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

    keys = keys.rstrip().split(delimiter)

    # task_id,num_gpus,submit_time,start_time,end_time,estimated_start_time,estimated_end_time,duration
    # file_data = file_data_filter(file_data, keys.index("duration"), operator.le, 200)
    # file_data = file_data_filter(file_data, keys.index("estimation_case"), operator.eq, 3.0)


    file_data = sorted(file_data, key=lambda e: int(e.rstrip().split(delimiter)[0]))


    for i, key in enumerate(keys):
        if "status" in key:
            continue
        result_dictionary[key] = list(map(lambda e: float(e.rstrip().split(delimiter)[i]), file_data))

    return result_dictionary

def cdf_queuing_delay(result_dictionary):
    queueing_delay = np.subtract(result_dictionary["start_time"], result_dictionary["submit_time"])
    queueing_delay.sort()
    cdf = np.linspace(1.0/len(queueing_delay),1.0,len(queueing_delay))
    return queueing_delay, cdf



def cdf_ACT(result_dictionary):
    ACT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    ACT.sort()
    cdf = np.linspace(1.0/len(ACT),1.0,len(ACT))
    return ACT, cdf


def cdf_ACT_error(result_dictionary, fname=None):


    actual_ACT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    estimated_ACT = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["submit_time"])
    estimated_execution = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["estimated_start_time"])

    ACT_error = list()
    for i in range(len(estimated_ACT)):

        if estimated_execution[i] != 0:
            ACT_error.append(100.0 * abs(estimated_ACT[i] - actual_ACT[i]) / estimated_ACT[i])
        # ACT_error.append(100.0 * abs(estimated_ACT[i] - actual_ACT[i]) / actual_ACT[i])
        # ACT_error.append(abs(estimated_ACT[i]) / actual_ACT[i])
        # ACT_error.append(abs(actual_ACT[i]) / estimated_ACT[i])
        # ACT_error.append(abs(estimated_ACT[i] - actual_ACT[i]))
        # ACT_error.append((estimated_ACT[i] - actual_ACT[i]))


    if len(ACT_error) == 0:
        ACT_error.append(0)
        ACT_error.append(0)

    '''
    for i in range(len(ACT_error)):
        if "FIFO" in fname and ACT_error[i] > 0:
            print("global_id: %d task.task_id: %d error: %f" % (i, result_dictionary["task_id"][i], ACT_error[i]))
    '''


    ACT_error.sort()
    cdf = np.linspace(1.0/len(ACT_error),1.0,len(ACT_error))
    return ACT_error, cdf


def cdf_execution_time(result_dictionary):
    execution_time = np.subtract(result_dictionary["end_time"], result_dictionary["start_time"])
    execution_time.sort()
    cdf = np.linspace(1.0/len(execution_time),1.0,len(execution_time))
    return execution_time, cdf



def cdf_execution_slowdown(result_dictionary):
    execution_time = np.subtract(result_dictionary["end_time"], result_dictionary["start_time"])
    duration = result_dictionary["duration"]


    execution_slowdown = np.divide(duration,execution_time)

    execution_slowdown.sort()
    cdf = np.linspace(1.0/len(execution_slowdown),1.0,len(execution_slowdown))
    return execution_slowdown, cdf

def cdf_execution_slowdown_uncertainty(result_dictionary):
    execution_time = np.subtract(result_dictionary["end_time"], result_dictionary["uncertainty_time"])
    duration = result_dictionary["remaining_task_duration"]

    execution_slowdown_uncertainty = np.divide(duration,execution_time)

    execution_slowdown_uncertainty.sort()
    cdf = np.linspace(1.0/len(execution_slowdown_uncertainty),1.0,len(execution_slowdown_uncertainty))
    return execution_slowdown_uncertainty, cdf


def cdf_slowdown(result_dictionary):
    ACT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    execution_time = np.subtract(result_dictionary["end_time"], result_dictionary["start_time"])

    slowdown = np.divide(ACT, execution_time)

    slowdown.sort()
    cdf = np.linspace(1.0/len(slowdown),1.0,len(slowdown))
    return slowdown, cdf



def cdf_queueing_delay_error(result_dictionary):

    actual_queueing_delay = np.subtract(result_dictionary["start_time"], result_dictionary["submit_time"])
    estimated_queueing_delay = np.subtract(result_dictionary["estimated_start_time"], result_dictionary["submit_time"])


    queueing_delay_error = list()
    for i in range(len(actual_queueing_delay)):

        if estimated_queueing_delay[i] > 0:
            queueing_delay_error.append(100.0 * abs(estimated_queueing_delay[i] - actual_queueing_delay[i]) / estimated_queueing_delay[i])
        else:
            queueing_delay_error.append(0)


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
    # ml_app_time = np.subtract(result_dictionary["end_time"], result_dictionary["start_time"])
    ml_app_time = result_dictionary["duration"]
    ml_app_time.sort()
    cdf = np.linspace(1.0/len(ml_app_time),1.0,len(ml_app_time))
    return ml_app_time, cdf


def cdf_app_service(result_dictionary):
    service = result_dictionary["service"]
    service.sort()
    cdf = np.linspace(1.0/len(service),1.0,len(service))
    return service, cdf

def cdf_app_budget(result_dictionary):
    budget = result_dictionary["budget"]
    budget.sort()
    cdf = np.linspace(1.0/len(budget),1.0,len(budget))
    return budget, cdf

def cdf_app_service_budget_error(result_dictionary):
    service = result_dictionary["service"]
    budget = result_dictionary["budget"]
    
    error = np.divide(service, budget)
    error.sort()
    cdf = np.linspace(1.0/len(error),1.0,len(error))
    return error, cdf


def cdf_unfairness(result_dictionary, fair_result_dictionary):
    ACT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    fair_ACT = np.subtract(fair_result_dictionary["end_time"], fair_result_dictionary["submit_time"])

    unfairness = list()

    for i in range(len(ACT)):

        unfairness.append(max(0, 100.0*( (ACT[i]/fair_ACT[i]) - 1.0)))

    unfairness.sort()
    cdf = np.linspace(1.0/len(unfairness),1.0,len(unfairness))
    return unfairness, cdf

def cdf_num_apps_seen_diff(result_dictionary):
    num_apps_seen_diff = result_dictionary["num_apps_seen_diff"]
    num_apps_seen_diff.sort()
    cdf = np.linspace(1.0/len(num_apps_seen_diff),1.0,len(num_apps_seen_diff))
    return num_apps_seen_diff, cdf




def cdf_finish_time_fairness(result_dictionary):

    '''
    end_times = result_dictionary["end_time"]
    fair_end_times = result_dictionary["fair_end_time"]
    '''

    ACT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    fair_ACT = result_dictionary["fair_act"]

    ACT_fairness = list()

    for i in range(len(ACT)):
        ACT_fairness.append(100.0* max(0, ((ACT[i] - fair_ACT[i])/fair_ACT[i])))


    # ACT_fairness = np.multiply(100.0,np.max(0, np.divide(np.subtract(ACT, fair_ACT), fair_ACT)))

    # ACT_fairness = np.divide(ACT)

    ACT_fairness.sort()
    cdf = np.linspace(1.0/len(ACT_fairness),1.0,len(ACT_fairness))
    return ACT_fairness, cdf



def avg_ACT(result_dictionary):
    ACT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    return np.mean(ACT)

def avg_ACT_error(result_dictionary):


    ACT_error, *_ = cdf_ACT_error(result_dictionary)
    return np.mean(ACT_error)


    actual_ACT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    estimated_ACT = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["submit_time"])
    estimated_execution = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["estimated_start_time"])

    ACT_error = list()
    for i in range(len(estimated_ACT)):
        if estimated_execution[i] == 0:
            ACT_error.append(0)
        else:
            ACT_error.append(100.0 * abs(estimated_ACT[i] - actual_ACT[i]) / estimated_ACT[i])

    return np.mean(ACT_error)


def avg_unfairness(result_dictionary, fair_result_dictionary):

    ACT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    fair_ACT = np.subtract(fair_result_dictionary["end_time"], fair_result_dictionary["submit_time"])

    unfairness = list()

    for i in range(len(ACT)):

        unfairness.append(max(0, 100.0*( (ACT[i]/fair_ACT[i]) - 1.0)))

    return np.mean(unfairness)

def avg_unfairness2(result_dictionary):

    ACT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    fair_ACT = result_dictionary["fair_act"]

    ACT_fairness = list()

    for i in range(len(ACT)):
        ACT_fairness.append(100.0* max(0, ((ACT[i] - fair_ACT[i])/fair_ACT[i])))


    # ACT_fairness = np.multiply(100.0,np.max(0, np.divide(np.subtract(ACT, fair_ACT), fair_ACT)))

    # ACT_fairness = np.divide(ACT)

    return np.mean(ACT_fairness)

def avg_execution_slowdown(result_dictionary):
    execution_time = np.subtract(result_dictionary["end_time"], result_dictionary["start_time"])
    duration = result_dictionary["duration"]


    execution_slowdown = np.divide(duration,execution_time)

    print(np.mean(execution_slowdown))

    return np.mean(execution_slowdown)
    
def predicted_rate(result_dictionary):
    duration = result_dictionary["duration"]
    estimated_execution = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["estimated_start_time"])
    rate = np.divide(duration, estimated_execution)
    index = list(range(len(duration)))
    return index, rate


def actual_rate(result_dictionary):
    duration = result_dictionary["duration"]
    actual_execution_time = np.subtract(result_dictionary["end_time"], result_dictionary["start_time"])
    rate = np.divide(duration, actual_execution_time)
    index = list(range(len(duration)))
    return index, rate

def pareto_avg_avg(result_dictionary):
    actual_ACT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    estimated_ACT = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["submit_time"])
    estimated_execution = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["estimated_start_time"])

    ACT_error = list()
    for i in range(len(estimated_ACT)):
        if estimated_execution[i] == 0:
            ACT_error.append(0)
        else:
            ACT_error.append(100.0 * abs(estimated_ACT[i] - actual_ACT[i]) / estimated_ACT[i])

    actual_ACT.sort()
    ACT_error.sort()

    perf_value = None
    pred_value = None

    return np.mean(ACT_error), np.mean(actual_ACT)


    # if perf_metric == "avg":
    #     perf_value = np.mean(ACT)
    # else:
    #     p = float(perf_metric.split("p")[-1])
    #     perf_value = np.percentile(ACT, p, interpolation="nearest")


    # if pred_metric == "avg":
    #     pred_value = np.mean(ACT_error)
    # else:
    #     p = float(pred_metric.split("p")[-1])
    #     pred_value = np.percentile(ACT_error, p, interpolation="nearest")

    # return perf_value, pred_value

def pareto_avg_tail(result_dictionary):
    actual_ACT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    estimated_ACT = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["submit_time"])
    estimated_execution = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["estimated_start_time"])

    ACT_error = list()
    for i in range(len(estimated_ACT)):
        if estimated_execution[i] == 0:
            ACT_error.append(0)
        else:
            ACT_error.append(100.0 * abs(estimated_ACT[i] - actual_ACT[i]) / estimated_ACT[i])

    actual_ACT.sort()
    ACT_error.sort()

    perf_value = None
    pred_value = None

    return np.percentile(ACT_error, 99, interpolation="nearest"), np.mean(actual_ACT)


def pareto_plotter_main(results, funcs):
    for func in funcs:

        fig, axs = plt.subplots(1,figsize=(8,4))
        # plt.figure()


        # fig.suptitle('Vertically stacked subplots')
        # axs[0].plot(x, y)
        # axs[1].plot(x, -y)

        # plt.figure()

        x_label=None
        ylabel = None
        x_lim = None
        y_lim = None
        x_ticks = None
        y_ticks=None

        X = list()
        Y = list()
        color_to_use = list()
        label_to_use = list()
        for i, fname in enumerate(results):


            # if any(list(map(lambda name: name in fname, func_to_exclude_fnames[func.__name__]))):
            #     continue

            x,y = func(results[fname])

            y = y/60.0

            X.append(x)
            Y.append(y)
            color_to_use.append(color_from_policies(policies[i]))
            label_to_use.append(label_from_policies(policies[i]))


        Y = list(map(lambda y: y/min(Y), Y))

        for i in range(len(X)):
            axs.scatter(X[i],Y[i],alpha=0.8,color=color_to_use[i], label=label_to_use[i],zorder=3, s=200)


        # axs.scatter(X,Y,alpha=[0.8]*len(X),color=color_to_use, label=label_to_use, zorder=3, s=[200]*len(X))

        
        if func.__name__ == "pareto_avg_avg":
            y_label = "Normalized Avg JCT"
            x_label = r"Avg $\mathcal{E}_{pred} \%$"
            
            



            # ypos, *_ = plt.xticks()

            # y_ticks = (ypos,
            #           list(map(lambda p: str(int(p/(60.0))), ypos)))
      
            # x_lim = (min(X)-5, max(X)+5)
            # y_ticks = ([200,400,600,800],list(map(str,[200,400,600,800])))


        set_canvas(axs, x_label=x_label, y_label=y_label, x_lim=x_lim, y_lim=y_lim, y_ticks=y_ticks, x_ticks=x_ticks, legend_loc='best')
        plt.savefig(f'{func.__name__}{output_identifier}.{args.format}', dpi = 300, bbox_inches='tight')

def cdf_plotter_main(results, funcs):


    func_to_exclude_fnames = {"cdf_avg_apps_seen": ["FIFO", "SRSF"]}


    fair_result_dictionary = None

    for fname in results:
        if "FS" in fname:
            print(fname)
            fair_result_dictionary = results[fname]
            break


    for func in funcs:


        # here
        plt.figure(figsize=(8,4))
        # fig, axs = plt.subplots(1)


        for i, fname in enumerate(fnames):
            result_dictionary = results[fname]

            # if fname in func_to_exclude_fnames[func.__name__]:
            #     continue


            if func.__name__ in func_to_exclude_fnames and any(list(map(lambda name: name in fname, func_to_exclude_fnames[func.__name__]))):
                continue

            if func.__name__ == "cdf_fairness" and fair_result_dictionary != None:
                x, y = func(result_dictionary, fair_result_dictionary)
            elif func.__name__ == "cdf_fairness" and fair_result_dictionary == None:
                continue
            else:    
                x, y = func(result_dictionary)

            
            label = label_from_policies(policies[i])
            

            if func.__name__ == "cdf_ACT":
                x = list(map(lambda p: p/60.0, x))

            plt.plot(x, y, color= color_from_policies(policies[i]), marker='o', markevery=markevery_calc(len(x)), label=label, linewidth=2)



        x_lim = None
        y_lim = None
        x_label = None
        y_label = "CDF"
        x_ticks = None
        y_ticks = None


        if func.__name__ == "cdf_avg_apps_seen":
            x_label = "Observed contention"
            plt.vlines(4, 0, 1, linewidth=2, linestyle="--", zorder=4, color='k', label="Avg contention")


        if "cdf_finish_time_fairness" == func.__name__:
            x_label = 'Finish Time Fairness'

        if "cdf_ACT" == func.__name__:
            plt.xscale('log')
            xlim = [1,100]
            x_ticks = [[1,10,100],
                        [1,10,100]]

            # x_ticks, *_ = plt.xticks()

            # x_ticks = [list(map(lambda t: t/60.0, x_ticks)),
            #         list(map(lambda t: str(int(t/60.0)), x_ticks))]


            print(x_ticks)

            x_label = 'JCT (minutes)'
            
        if "cdf_ACT_error" in func.__name__:
            x_lim = [0,300]
            # x_label = r"$\frac{|Estimate-Actual|}{Estimate}$ %"
            x_label = r"$\mathcal{E}_{pred} \%$"
            # plt.xscale('symlog')

        # y_ticks, *_ = plt.yticks()

        # y_ticks = [list(map(lambda yt: round(yt,1), y_ticks)),
        #             list(map(lambda yt: str(round(yt,1)), y_ticks))]

        y_ticks = [[0,0.2,0.4,0.6,0.8,1.0], list(map(str,[0,0.2,0.4,0.6,0.8,1.0]))]

        # ylim = [0,1]


        set_canvas(plt.gca(), x_label, y_label, x_lim=x_lim, y_lim=y_lim, y_ticks=y_ticks, x_ticks=x_ticks, legend_loc='best')
        plt.savefig(f'{func.__name__}{output_identifier}.{args.format}', dpi = 300, bbox_inches='tight')



def bars_ACT(result_dictionary):
    ACT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    ACT.sort()
    
    measures = {"Avg": np.mean(ACT),
                "p90": np.percentile(ACT, 90, interpolation='nearest'),
                "p95": np.percentile(ACT, 95, interpolation='nearest'),
                "p99": np.percentile(ACT, 99, interpolation='nearest')}
                # "p99.9": np.percentile(ACT, 99.9, interpolation='nearest')}


    measures = {"Avg": np.mean(ACT)}

    # measures = {"Avg": np.mean(ACT)}
    #             # "p99.9": np.percentile(ACT, 99.9, interpolation='nearest')}
				# # "p99": np.percentile(ACT, 99, interpolation='nearest')


    return measures

def bars_ACT_avg_error_avg(result_dictionary):
    actual_ACT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    estimated_ACT = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["submit_time"])
    estimated_execution = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["estimated_start_time"])

    ACT_error = list()
    for i in range(len(estimated_ACT)):
        if estimated_execution[i] == 0:
            ACT_error.append(0)
        else:
            ACT_error.append(100.0 * abs(estimated_ACT[i] - actual_ACT[i]) / estimated_ACT[i])

    ACT_error.sort()

    
    measures = {"Avg JCT (min)": np.mean(actual_ACT),
    			r"Avg $\mathcal{E}_{pred}$": np.mean(ACT_error),
                r"p99 $\mathcal{E}_{pred}$": np.percentile(actual_ACT, 99, interpolation='nearest')}
                # "p99.9": np.percentile(ACT_error, 99.9, interpolation='nearest')

    return measures




def bars_ACT_error(result_dictionary):

    actual_ACT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    estimated_ACT = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["submit_time"])
    estimated_execution = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["estimated_start_time"])

    ACT_error = list()
    for i in range(len(estimated_ACT)):
        if estimated_execution[i] == 0:
            ACT_error.append(0)
        else:
            ACT_error.append(100.0 * abs(estimated_ACT[i] - actual_ACT[i]) / estimated_ACT[i])

    ACT_error.sort()


    measures = {"Avg": np.mean(ACT_error),
                "p90": np.percentile(ACT_error, 90, interpolation='nearest'),
                "p95": np.percentile(ACT_error, 95, interpolation='nearest'),
                "p99": np.percentile(ACT_error, 99, interpolation='nearest')}

    measures = {"Avg": np.mean(ACT_error)}

    
    # measures = {"Avg": np.mean(ACT_error),
    #             "p99": np.percentile(ACT_error, 99, interpolation='nearest')}
    #             # "p99.9": np.percentile(ACT_error, 99.9, interpolation='nearest')

    return measures


def bars_unfairness(result_dictionary):


    unfairness, _ = cdf_finish_time_fairness(result_dictionary)

    measures = {"Avg": np.mean(unfairness),
                "p90": np.percentile(unfairness, 90, interpolation='nearest'),
                "p95": np.percentile(unfairness, 95, interpolation='nearest'),
                "p99": np.percentile(unfairness, 99, interpolation='nearest')}


    measures = {"max": max(unfairness)}

    
    # measures = {"Avg": np.mean(ACT_error),
    #             "p99": np.percentile(ACT_error, 99, interpolation='nearest')}
    #             # "p99.9": np.percentile(ACT_error, 99.9, interpolation='nearest')

    return measures



def bars_makespan(result_dictionary):
    return {"Makespan": max(result_dictionary["end_time"]) - min(result_dictionary["submit_time"])}


def violin_size_vs_error(result_dictionary):
    actual_ACT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    estimated_ACT = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["submit_time"])
    estimated_execution = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["estimated_start_time"])
    service = result_dictionary["service"]

    min_service, max_service = min(service), max(service)

    num_buckets = 5

    thresholds = np.linspace(min_service, max_service, num_buckets+1)[1:]

    bucket_to_error = [ [] for  _ in range(num_buckets)]

    for i in range(len(estimated_ACT)):
        for bucket in range(num_buckets):
            if service[i] <= thresholds[bucket]:
    

                if estimated_execution[i] == 0:
                        error = 0
                else:
                    error = 100.0 * abs(estimated_ACT[i] - actual_ACT[i]) / estimated_ACT[i]

                bucket_to_error[bucket].append(error)
                break


    return bucket_to_error, [min_service] + thresholds

    # print(map(sum, bucket))



    # ACT_error = list()
    # service = list()
    # for i in range(len(estimated_ACT)):

    
    #     # if np.random.uniform() < 1.0 and error > 50:
    #     if np.random.uniform() < 1.0:
    #         ACT_error.append(error)
    #         service.append(result_dictionary["service"][i])

    # return service, ACT_error


    


def scatter_size_vs_error(result_dictionary):
    actual_ACT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    estimated_ACT = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["submit_time"])
    estimated_execution = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["estimated_start_time"])


    ACT_error = list()
    service = list()
    for i in range(len(estimated_ACT)):

        if estimated_execution[i] == 0:
            error = 0
        else:
            error = 100.0 * abs(estimated_ACT[i] - actual_ACT[i]) / estimated_ACT[i]

        # if np.random.uniform() < 1.0 and error > 50:
        if np.random.uniform() < 1.0:
            ACT_error.append(error)
            service.append(result_dictionary["service"][i])

    return service, ACT_error


def scatter_slowdown_vs_error(result_dictionary):
    actual_ACT = np.subtract(result_dictionary["end_time"], result_dictionary["submit_time"])
    estimated_ACT = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["submit_time"])
    slowdown = np.divide(actual_ACT, result_dictionary["service"])
    estimated_execution = np.subtract(result_dictionary["estimated_end_time"], result_dictionary["estimated_start_time"])

    ACT_error = list()
    X = list()
    for i in range(len(estimated_ACT)):

        if estimated_execution[i] == 0:
            error = 0
        else:
            error = 100.0 * abs(estimated_ACT[i] - actual_ACT[i]) / estimated_ACT[i]

        # if np.random.uniform() < 1.0 and error > 50:
        if np.random.uniform() < 1.0:
            ACT_error.append(error)
            X.append(slowdown[i])
            

    return X, ACT_error



def cdf_avg_apps_seen(result_dictionary):
    
    avg_apps_seen = result_dictionary["num_apps_seen_diff"]
    avg_apps_seen.sort()
    
    print(f"avg of avg_apps_seen: {np.mean(avg_apps_seen)}")

    cdf = np.linspace(0,1,len(avg_apps_seen))
    return avg_apps_seen, cdf


def scatter_app_id_vs_avg_apps_seen(result_dictionary):
    
    app_ids = result_dictionary["app_id"]
    avg_apps_seen = result_dictionary["num_apps_seen_diff"]

    return app_ids, avg_apps_seen




def scatter_plotter_main(results, funcs):

    func_to_exclude_fnames = {"scatter_size_vs_error": ["FIFO", "FS"],
                            "scatter_app_id_vs_avg_apps_seen": ["FIFO", "SRSF"]}

    for func in funcs:



        fig, axs = plt.subplots(1)
        # fig.suptitle('Vertically stacked subplots')
        # axs[0].plot(x, y)
        # axs[1].plot(x, -y)

        # plt.figure()

        x_label=None
        ylabel = None
        x_lim = None
        y_lim = None
        x_ticks = None
        y_ticks=None

        for i, fname in enumerate(results):


            if any(list(map(lambda name: name in fname, func_to_exclude_fnames[func.__name__]))):
                continue
            X,Y = func(results[fname])


            axs.scatter(X,Y,alpha=0.7,color=color_from_policies(policies[i]), label=label_from_policies(policies[i]), zorder=3)


        if func.__name__ == "scatter_size_vs_error":
            y_label = r"$\mathcal{E}_{pred} \%$"
            x_label = "Size (hrs)"
            xpos, *_ = plt.xticks()

            x_ticks = (xpos,
                      list(map(lambda p: str(int(p/(60.0))), xpos)))
      
            x_lim = (min(X)-5, max(X)+5)
            y_ticks = ([200,400,600,800],list(map(str,[200,400,600,800])))

        if func.__name__ == "scatter_app_id_vs_avg_apps_seen":
            y_label = "Observed contention"
            x_label = "Job ID"
            axs.hlines(4, min(X), max(X), linewidth=2, linestyle="--", zorder=4, color='k', label="Average contention")

        print(x_lim)

        set_canvas(axs, x_label=x_label, y_label=y_label, x_lim=x_lim, y_lim=y_lim, y_ticks=y_ticks, x_ticks=x_ticks, legend_loc='best')
        plt.savefig(f'{func.__name__}{output_identifier}.{args.format}', dpi = 300, bbox_inches='tight')



def bar_plotter_main(results, funcs):


    barwidth=1.0/(len(results)+1)
    

    for func in funcs:



        plt.figure()

        x_label=None
        y_label = None
        x_lim = None
        y_lim = None
        x_ticks = None
        y_ticks=None



        fname_to_bar_data = {}


        for i, fname in enumerate(results):
            fname_to_bar_data[fname] = func(results[fname])

            xplaces = [x + (barwidth*i) for x in np.arange(len(fname_to_bar_data[fname]))]


            if func.__name__ == "bars_ACT_error":
                for key in fname_to_bar_data[fname]:
                    fname_to_bar_data[fname][key] += 5




            plt.bar(xplaces, fname_to_bar_data[fname].values(),
                    color=color_from_policies(policies[i]),
                    label=label_from_policies(policies[i]),
                    width=barwidth,
                    zorder=3)

        if func.__name__ == "bars_ACT":
            y_label = "JCTs (hrs)"
            ypos, *_ = plt.yticks()

            y_ticks = (ypos,
                      list(map(lambda p: str(int(p/(60.0))), ypos)))

        elif func.__name__ == "bars_ACT_error":
            y_label = r"$\mathcal{E}_{pred} \%$"


            ylim = [-2,500]

            ypos, *_ = plt.yticks()

            ypos = ypos[:-1]

            y_ticks = (list(map(lambda yt: yt+5, ypos)),
                      list(map(lambda yt: str(int(yt)), ypos)))


            

        x_ticks = ([r + barwidth for r in range(len(fname_to_bar_data[fname]))],
                    fname_to_bar_data[fname].keys())
        
        set_canvas(plt.gca(), x_label=x_label, y_label=y_label, x_lim=x_lim, y_lim=y_lim, y_ticks=y_ticks, x_ticks=x_ticks, legend_loc='best')
        plt.savefig(f'{func.__name__}{output_identifier}.{args.format}', dpi = 300, bbox_inches='tight')

def average_plotter_main(results, funcs):
    fair_result_dictionary = None

    for fname in results:
        if "FS" in fname:
            print(fname)
            fair_result_dictionary = results[fname]
            break
    
    for i, func in enumerate(funcs):
        X = list()
        Y = list()
        for fname in results:
            result_dictionary = results[fname]

            if func.__name__ == "avg_unfairness" and fair_result_dictionary != None:
                Y.append(func(result_dictionary, fair_result_dictionary))
            elif func.__name__ != "avg_unfairness":
                Y.append(func(result_dictionary))

            label = label_from_policies(policies[i])

            # fname.split('/')[-1].split('.')[0] #.split('_')[-2]
            X.append(label)

        # Y = list(map(lambda y: y/min(Y), Y))
        plt.bar(X,Y,color=colors[:len(X)])
        plt.xlabel('Sched policies')
        # plt.ylabel(func.__name__.split('_')[1])
        plt.ylabel(func.__name__)

        plt.grid(alpha=.8, linestyle='--')
        plt.savefig(f'{func.__name__}{output_identifier}.{args.format}', dpi = 300)
        # plt.figure()

def violin_plotter_main(results, funcs):
    for func in funcs:



        x_label=None
        ylabel = None
        x_lim = None
        y_lim = None
        x_ticks = None
        y_ticks=None


        for i, fname in enumerate(results):

            fig, axs = plt.subplots(1)

            buckets, groups = func(results[fname])

            # xplaces = [x + (barwidth*i) for x in np.arange(len(fname_to_bar_data[fname]))]


            # if func.__name__ == "bars_ACT_error":
            #     for key in fname_to_bar_data[fname]:
            #         fname_to_bar_data[fname][key] += 5



            print(groups)

            violin_parts = axs.violinplot(buckets, groups, showmeans=False, widths=1000, showextrema=False)

            axs.legend([violin_parts], ['label1'])

            for part in violin_parts['bodies']:
                part.set_facecolor(color_from_policies(policies[i]))
                part.set_edgecolor(color_from_policies(policies[i]))
                part.set_alpha(0.7)


            if func.__name__ == "violin_size_vs_error":
                xlabel = "Job sizes"
                y_label = r"$\mathcal{E}_{pred} \%$"



            # x_ticks = ([r + barwidth for r in range(len(fname_to_bar_data[fname]))],
            #             fname_to_bar_data[fname].keys())
            
            set_canvas(axs, x_label=x_label, y_label=y_label, x_lim=x_lim, y_lim=y_lim, y_ticks=y_ticks, x_ticks=x_ticks, legend_loc='best', showgrid=False)
            plt.savefig(f'{func.__name__}_{policies[i]}{output_identifier}.{args.format}', dpi = 300, bbox_inches='tight')

            # if func.__name__ == "bars_ACT":
            #     y_label = "JCTs (hrs)"
            #     ypos, *_ = plt.yticks()

            #     y_ticks = (ypos,
            #               list(map(lambda p: str(int(p/(60.0))), ypos)))

            # elif func.__name__ == "bars_ACT_error":
            #     y_label = r"$\mathcal{E}_{CTE} \%$"


            #     ylim = [-2,500]

            #     ypos, *_ = plt.yticks()

            #     ypos = ypos[:-1]

            #     y_ticks = (list(map(lambda yt: yt+5, ypos)),
            #               list(map(lambda yt: str(int(yt)), ypos)))




def seed_bar_plotter_main(results, funcs):


    
    

    num_seeds = int(len(policies)/len(set(policies)))

    barwidth=1.0/(len(set(policies))+1)

    for func in funcs:



        plt.figure()

        x_label=None
        y_label = None
        x_lim = None
        y_lim = None
        x_ticks = None
        y_ticks=None



        fname_to_bar_data = {}

        group = []


        seed = 0

        for i, fname in enumerate(results):
            fname_to_bar_data[fname] = func(results[fname])


            if len(group) == 0:
                xplaces = [x + (barwidth*seed) for x in np.arange(len(fname_to_bar_data[fname]))]


            if func.__name__ == "bars_ACT_error":
                for key in fname_to_bar_data[fname]:
                    fname_to_bar_data[fname][key] += 5


            group.append(list(fname_to_bar_data[fname].values()))

            if len(group) == num_seeds:
                
                bar_heights = np.mean(group, axis=0)

                plt.bar(xplaces, bar_heights,
                        color=color_from_policies(policies[i]),
                        label=label_from_policies(policies[i]),
                        width=barwidth,
                        zorder=3)
                group = []
                seed += 1

        if func.__name__ == "bars_ACT":
            y_label = "JCTs (min)"
            ypos, *_ = plt.yticks()

            y_ticks = (ypos,
                      list(map(lambda p: str(int(p/(60.0))), ypos)))

        elif func.__name__ == "bars_ACT_error":
            y_label = r"$\mathcal{E}_{pred} \%$"


            ylim = [-2,500]

            ypos, *_ = plt.yticks()

            ypos = ypos[:-1]

            y_ticks = (list(map(lambda yt: yt+5, ypos)),
                      list(map(lambda yt: str(int(yt)), ypos)))


            

        x_ticks = ([r + barwidth for r in range(len(fname_to_bar_data[fname]))],
                    fname_to_bar_data[fname].keys())
        
        set_canvas(plt.gca(), x_label=x_label, y_label=y_label, x_lim=x_lim, y_lim=y_lim, y_ticks=y_ticks, x_ticks=x_ticks, legend_loc='best')
        plt.savefig(f'{func.__name__}{output_identifier}.{args.format}', dpi = 300, bbox_inches='tight')


def seed_bar_plotter_main_single(results, funcs):


    
    

    num_seeds = int(len(policies)/len(set(policies)))

    barwidth=1.0/(len(set(policies))+1)

    sep = 1.5

    for func in funcs:



        plt.figure()

        x_label=None
        y_label = None
        x_lim = None
        y_lim = None
        x_ticks = None
        y_ticks=None



        fname_to_bar_data = {}

        group = []


        seed = 0

        for i, fname in enumerate(results):
            fname_to_bar_data[fname] = func(results[fname])


            if len(group) == 0:
                xplaces = [seed*sep*barwidth]
                # [x + (barwidth*seed) for x in np.arange(len(fname_to_bar_data[fname]))]


            if func.__name__ == "bars_ACT_error":
                for key in fname_to_bar_data[fname]:
                    fname_to_bar_data[fname][key] += 5


            group.append(list(fname_to_bar_data[fname].values()))

            if len(group) == num_seeds:
                
                bar_heights = np.mean(group, axis=0)


                plt.bar(xplaces, bar_heights,
                        color=color_from_policies(policies[i]),
                        width=barwidth,
                        zorder=3)
                group = []
                seed += 1

        if func.__name__ == "bars_ACT":
            y_label = "Avg JCTs (hrs)"
            ypos, *_ = plt.yticks()

            y_ticks = (ypos,
                      list(map(lambda p: str(int(p/(60.0))), ypos)))



        elif func.__name__ == "bars_ACT_error":
            y_label = "Avg "+r"$\mathcal{E}_{pred} \%$"


            ylim = [-2,500]

            ypos, *_ = plt.yticks()

            ypos = ypos[:-1]

            y_ticks = (list(map(lambda yt: yt+5, ypos)),
                      list(map(lambda yt: str(int(yt)), ypos)))


            

        if func.__name__ == "bars_unfairness":
            y_label = r"Max Unfairness $\%$"
            y_lim = [0,500]
            # y_ticks = [[0,100,200,300,400,500],
            # ["0","100","200","300","400","500"]]

        x_ticks = ([i*sep*barwidth for i in range(len(set(policies)))], [label_from_policies(policies[i]) for i in range(0,len(policies),num_seeds)])
        

        set_canvas(plt.gca(), x_label=x_label, y_label=y_label, x_lim=x_lim, y_lim=y_lim, y_ticks=y_ticks, x_ticks=x_ticks, legend_loc='best', showlegend=False)
        plt.savefig(f'{func.__name__}{output_identifier}.{args.format}', dpi = 300, bbox_inches='tight')




def seed_cdf_plotter(results, funcs):
    func_to_exclude_fnames = {"cdf_avg_apps_seen": ["FIFO", "SRSF"]}

    

    num_seeds = int(len(policies)/len(set(policies)))

    print(num_seeds)

    for func in funcs:


        # here
        # plt.figure()
        fig, axs = plt.subplots(1, figsize=(8,4))

        group = []

        for i, fname in enumerate(fnames):

            result_dictionary = results[fname]

            # if fname in func_to_exclude_fnames[func.__name__]:
            #     continue


            if func.__name__ in func_to_exclude_fnames and any(list(map(lambda name: name in fname, func_to_exclude_fnames[func.__name__]))):
                continue

            x, y = func(result_dictionary)

            if func.__name__ in ["cdf_ACT"] and "gavel" not in fname:
                x = list(map(lambda p: p/60.0, x))

            group.append([x,y])

            if len(group) == num_seeds:

                xs = [group[seed][0] for seed in range(num_seeds)]
                # mean for each percentile
                x = np.mean(xs, axis=0)
                y = group[0][1]

                axs.plot(x, y, color= color_from_policies(policies[i]), marker=markers[get_index_policy(policies[i])],
                        markevery=markevery_calc(len(x)), label=label_from_policies(policies[i]), linewidth=2, linestyle=linestyles[get_index_policy(policies[i])])
                group = []

    


        x_lim = None
        y_lim = None
        x_label = None
        y_label = "CDF"
        x_ticks = None
        y_ticks = None


        if func.__name__ == "cdf_avg_apps_seen":
            x_label = "Observed contention"
            axs.vlines(4, 0, 1, linewidth=2, linestyle="--", zorder=4, color='k', label="Avg contention")


        if "cdf_finish_time_fairness" == func.__name__:
            x_label = r'$\rho{}$ %'
            x_lim = [0,200]

        if "cdf_ACT" == func.__name__:
            plt.xscale('log')

            '''
            xlim = [1,100]
            x_ticks = [[1,10,100],
                        [1,10,100]]
            '''

            # x_ticks, *_ = plt.xticks()

            # x_ticks = [list(map(lambda t: t/60.0, x_ticks)),
            #         list(map(lambda t: str(int(t/60.0)), x_ticks))]


            print(x_ticks)

            x_label = 'JCT (minutes)'
            
        if "cdf_ACT_error" in func.__name__:
            x_lim = [0,300]
            # x_label = r"$\frac{|Estimate-Actual|}{Estimate}$ %"
            x_label = r"$\mathcal{E}_{pred} \%$"
            # plt.xscale('symlog')

        # y_ticks, *_ = plt.yticks()

        # y_ticks = [list(map(lambda yt: round(yt,1), y_ticks)),
        #             list(map(lambda yt: str(round(yt,1)), y_ticks))]

        y_ticks = [[0,0.2,0.4,0.6,0.8,1.0], list(map(str,[0,0.2,0.4,0.6,0.8,1.0]))]

        # ylim = [0,1]


        set_canvas(axs, x_label, y_label, x_lim=x_lim, y_lim=y_lim, y_ticks=y_ticks, x_ticks=x_ticks, legend_loc='best')
        plt.savefig(f'{func.__name__}{output_identifier}.{args.format}', dpi = 300, bbox_inches='tight')

            
def seed_pareto_plotter_main(results, funcs):
    
    num_seeds = int(len(policies)/len(set(policies)))


    for func in funcs:

        fig, axs = plt.subplots(1,figsize=(8,4))
        # plt.figure()


        # fig.suptitle('Vertically stacked subplots')
        # axs[0].plot(x, y)
        # axs[1].plot(x, -y)

        # plt.figure()

        x_label=None
        y_label = None
        x_lim = None
        y_lim = None
        x_ticks = None
        y_ticks=None

        X = list()
        Y = list()
        color_to_use = list()
        label_to_use = list()

        group = list()
        seed = 0

        X = list()
        Y = list()

        for i, fname in enumerate(results):


            # if any(list(map(lambda name: name in fname, func_to_exclude_fnames[func.__name__]))):
            #     continue

            x,y = func(results[fname])

            if "gavel" not in fname:
                y = y/60.0


            group.append([x,y])

            if len(group) == num_seeds:

                x = np.mean([group[seed][0] for seed in range(num_seeds)])
                y = np.mean([group[seed][1] for seed in range(num_seeds)])

                print(f"scheme: {policies[i]}")
                print(f"avg error: {x}")
                print(f"avg jct: {y}")

                X.append(x)
                Y.append(y)
            
                color_to_use.append(color_from_policies(policies[i]))
                label_to_use.append(label_from_policies(policies[i]))

                group = list()
                seed += 1


        # axs.scatter(X,Y,alpha=[0.8]*len(X),color=color_to_use, label=label_to_use, zorder=3, s=[200]*len(X))



        # Y = list(map(lambda y: y/60.0, Y))        
        Y = list(map(lambda y: y/min(Y), Y))

        for i in range(len(X)):
            axs.scatter(X[i],Y[i],alpha=0.8,color=color_to_use[i], label=label_to_use[i], zorder=3, s=200)                    


        if func.__name__ == "pareto_avg_avg":
            y_label = "Normalized Avg JCT"
            x_label = r"Avg $\mathcal{E}_{pred} \%$"

        if func.__name__ == "pareto_avg_tail":
            y_label = "Normalized Avg JCT"
            x_label = r"p99 $\mathcal{E}_{pred} \%$"

            
            # ypos, *_ = plt.xticks()

            # y_ticks = (ypos,
            #           list(map(lambda p: str(int(p/(60.0))), ypos)))
      
            # x_lim = (min(X)-5, max(X)+5)
            # y_ticks = ([200,400,600,800],list(map(str,[200,400,600,800])))


        set_canvas(axs, x_label=x_label, y_label=y_label, x_lim=x_lim, y_lim=y_lim, y_ticks=y_ticks, x_ticks=x_ticks, legend_loc='best',legend_ncol=1)
        plt.savefig(f'{func.__name__}{output_identifier}.{args.format}', dpi = 300, bbox_inches='tight')






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fnames', nargs="+", help = "input filenames", type=str,required=True)
    parser.add_argument('-policies',nargs="+", help="policy names", type=str, required=True)
    parser.add_argument('-format', help= "output format (png or pdf)", type=str, default="png", choices=["png", "pdf"])
    parser.add_argument('-output_identifier', help= "output identifier", type=str, default=None)



    args = parser.parse_args()

    policies = args.policies

    fnames = args.fnames

    if args.output_identifier == None:
        output_identifier = ""
    else:
        output_identifier = f"_{args.output_identifier}"



    # colors = ['b','r','g','k','y','c','darkviolet'] + ['gainsboro','lightcoral','peru','forestgreen','lightgreen','paleturquoise','steelblue','lightsteelblue','blue','mediumorchid','plum','lightpink']
    

    results = {}
    for fname in fnames:

        results[fname] = parse_file_data(fname)

        # print(avg_ACT(results[fname]))



    seed_cdf_plotter(results, [cdf_ACT_error, cdf_ACT])
    # seed_cdf_plotter(results, [cdf_ACT_error, cdf_ACT, cdf_finish_time_fairness])

    # seed_pareto_plotter_main(results, [pareto_avg_avg])

    # seed_bar_plotter_main(results, [bars_ACT_error, bars_ACT])
    # seed_bar_plotter_main_single(results, [bars_ACT, bars_ACT_error, bars_unfairness])

    # seed_pareto_plotter_main(results, [pareto_avg_tail])

    # violin_plotter_main(results, [violin_size_vs_error])


    # average_plotter_main(results, [avg_ACT, avg_ACT_error])
    # cdf_plotter_main(results, [cdf_ACT, cdf_ACT_error,cdf_finish_time_fairness])

    # seed_bar_plotter_main(results, [bars_ACT, bars_ACT_error])
    # seed_bar_plotter_main(results, [bars_ACT_error])
    # pareto_plotter_main(results, [pareto_avg_avg])
    # cdf_plotter_main(results, [cdf_ACT, cdf_ACT_error, cdf_finish_time_fairness])

    # scatter_plotter_main(results, [scatter_size_vs_error])

    # cdf_plotter_main(results, [cdf_avg_apps_seen])
    # cdf_plotter_main(results, [cdf_ACT])
    # cdf_plotter_main(results, [cdf_ACT_error, cdf_ACT, cdf_context, cdf_execution_slowdown])
    # series_plotter(results)





