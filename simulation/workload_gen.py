import os, sys, argparse
import numpy as np
import csv
import copy, bisect
import matplotlib.pyplot as plt
import pickle
from pprint import pprint

from MCSScheduler import AppMCScheduler, AppPracticalMCScheduler
from PriorityScheduler import AppPrioScheduler
from FairScheduler import AppFairScheduler, AppPracticalFairScheduler
from ThemisScheduler import AppThemisScheduler
from AFSScheduler import AppAFSScheduler

from helpers import gen_data_from_cdf

from common import App, Job, Event

from datetime import datetime, timedelta

from functools import partial
from fractions import Fraction as frac

from models import Models



WORKLOADS = ["workload_0e4a51",
    "workload_b436b2",
    "workload_e13805",
    "workload_7f04ca",
    "workload_103959",
    "workload_ee9e8c",
    "workload_2869ce",
    "workload_ed69ec",
    "workload_11cb48",
    "workload_51b7ef",
    "workload_6214e9",
    "workload_6c71a0",
    "afs",
    "themis1",
    "themis2",
    "themis-scaled-2",
    "gavel"]

TRACES = ['trace_6c71a0',
        'trace_ee9e8c',
        'trace_103959',
        'trace_ed69ec',
        'trace_6214e9',
        'trace_7f04ca',
        'trace_b436b2',
        'trace_0e4a51',
        'trace_e13805',
        'trace_51b7ef',
        'trace_11cb48',
        'trace_2869ce']


def generate_SHA_jobs(app_id, num_jobs, service):

    jobs = {}

    alpha = 2
    total_stages = int(np.floor(np.log(num_jobs)/np.log(alpha))) + 1
    active_jobs = [int(np.floor(num_jobs/np.power(alpha, stage))) for stage in range(total_stages)]
    jobs_per_stage = [active_jobs[stage-1] - active_jobs[stage] for stage in range(1,total_stages)] + [1]
    fraction_per_stage = [np.power(alpha, stage) for stage in range(total_stages)]
    tau = service/np.dot(jobs_per_stage, fraction_per_stage)


    '''
    print(np.dot(jobs_per_stage, fraction_per_stage))
    print(jobs_per_stage)
    print(fraction_per_stage)
    print("=================")
    '''

    service_per_stage = np.multiply(tau, fraction_per_stage)
    jobs_per_stage = list(np.cumsum(jobs_per_stage))

    stage = 0
    for job_id in range(num_jobs):

        if job_id == jobs_per_stage[stage]:
            stage += 1

        jobs[job_id] = Job(app_id=app_id, job_id=job_id, service=service_per_stage[stage],
                            demand=1,
                            min_demand=1)

    return jobs


def generate_rect_jobs(app_id, num_jobs, service, max_gpus_per_job, min_gpus_per_job):
    jobs = {}

    for job_id in range(num_jobs):
        job = Job(app_id=app_id, job_id=job_id, service= (service/num_jobs),
                            demand=np.random.choice(max_gpus_per_job),
                            min_demand=np.random.choice(min_gpus_per_job))    
        

        raise NotImplementedError
        job.thrpt_dic = gen_thrpt_dic(job)


        jobs[job_id] = job

    return jobs


def generate_1_job(app_id, num_jobs, service, max_gpus_per_job, models, model_name=None):
    

    constrain_demand = False

    demand = np.sum(np.random.choice(max_gpus_per_job, num_jobs))

    jobs = {}
    job_id = 0

    if model_name == None:
        model = models.pick_random_model(max_gpus=demand)
    else:
        model = models.pick_model_by_name(model_name)
    # model = all_models.choices[app_id % 2]

    # print(f"model.name: {model.name}\tunconstrained_demand: {demand}")


    '''
    if constrain_demand:

        model_labels = ["Vgg", "Google", "Inception", "Resnet",
                        "Dcgan", "Video", "Chat", "Deep", "Transformer"]
        max_demands = [8, 10, 52, 10, 8, 4, 2, 4, 8]

        for i, label in enumerate(model_labels):
            if label in model.name:
                demand = min(max_demands[i], demand)
                break
    '''

    job = Job(app_id=app_id, job_id=job_id, service=service,
                            demand=demand,
                            min_demand=0)    

    job.thrpt_dic = [0.0] + model.speedups

    jobs[job_id] = job

    return jobs


def gen_workload_from_trace(trace_name, app_list, event_queue, models, max_apps=float('inf')):
    # app trace should follow app_id,total_stages,submit_time,job_id,num_gpu,_,stage_id,_,duration,deadline format
    # app list is a dictionary mapping from app_id to object App
    

    file_dir = os.path.dirname(os.path.abspath(__file__))

    submit_time = datetime.now()


    ###############################################################################
    # This previously used fname in place of trace_name which was a variable that
    # did not exist. 
    # 11/29/23
    ###############################################################################
    with open(f"{file_dir}/traces/{trace_name}.csv", 'r') as fp:
        csvReader = csv.reader(fp)
        next(csvReader)
        for app_id, row in enumerate(csvReader):


            # print(row)
            if "#" in "".join(row) or app_id > max_apps:

                continue



            _,_,service,num_jobs,sleep_time,*model_name = row

            app_id = int(app_id)
            service = float(service)
            num_jobs = int(num_jobs)
            sleep_time = float(sleep_time)


            jobs = generate_1_job(app_id, num_jobs, service, [1], models, model_name[0] if model_name else None)
            

            # jobs = generate_rect_jobs(app_id, num_jobs, service, [1], [1])
            


            # jobs = generate_SHA_jobs(app_id, num_jobs, service)


            app = App(app_id=app_id, jobs=jobs, deadline=None)

            app_list[app.app_id] = app

            submit_time += timedelta(seconds=sleep_time)

            event = Event(event_id=app_id, event_time=submit_time, event_type=Event.APP_SUB, app_id=app_id)
            

            event_queue.append(event)

        print("%d Apps generated" % (len(event_queue)))
        event_queue.reverse()



def gen_workload(cdf_app_service_times,
                cdf_num_jobs_per_app,
                cdf_max_gpus_per_job,
                cdf_min_gpus_per_job,
                load, num_gpus, num_apps, seed, app_list, event_queue, models):


    np.random.seed(seed)


    file_dir = os.path.dirname(os.path.abspath(__file__))



    app_service_times = gen_data_from_cdf(f"{file_dir}/workloads/cdf-app-service-times-{cdf_app_service_times}.csv",
                                        num_points=num_apps, dtype=int, interpolation=True)


    num_jobs_per_app = gen_data_from_cdf(f"{file_dir}/workloads/cdf-num-jobs-per-app-{cdf_num_jobs_per_app}.csv",
                                        num_points=num_apps, dtype=int, interpolation=True)


    max_gpus_per_job = gen_data_from_cdf(f"{file_dir}/workloads/cdf-max-gpus-per-job-{cdf_max_gpus_per_job}.csv",
                                        num_points=100, dtype=int, interpolation=True)


    min_gpus_per_job = gen_data_from_cdf(f"{file_dir}/workloads/cdf-min-gpus-per-job-{cdf_min_gpus_per_job}.csv",
                                        num_points=100, dtype=int, interpolation=True)


    
    interarrival_fname = f"{file_dir}/workloads/cdf-app-interarrival-times-{cdf_app_service_times}.csv"
    if os.path.exists(interarrival_fname):
        inter_arrival_times = gen_data_from_cdf(interarrival_fname,
                                            num_points=num_apps, dtype=int, interpolation=True)
        avg_interarrival_time = np.mean(inter_arrival_times)
        load = (np.mean(app_service_times))/((avg_interarrival_time)*num_gpus)
    else:
        avg_interarrival_time = (np.mean(app_service_times))/((load)*num_gpus)
        inter_arrival_times = np.random.exponential(avg_interarrival_time, size=(num_apps))

    print(f"avg_interarrival_time: {avg_interarrival_time}")
    print(f"load: {load}")
    print(f"avg_service_time: {np.mean(app_service_times)}")
    print(f"apps per min: {60.0/avg_interarrival_time}")


    start_time = datetime.now()
    submit_time = datetime.now()
    sleep_time = 0



    with open("gen_trace.csv",'w') as fp:

        for app_id in range(num_apps):

            num_jobs = num_jobs_per_app[app_id]
            service = max(int(float(app_service_times[app_id])/num_jobs), 1)*num_jobs

            # jobs = generate_rect_jobs(app_id, num_jobs, service, max_gpus_per_job, min_gpus_per_job)
            jobs = generate_1_job(app_id, num_jobs, service, max_gpus_per_job, models)

            app = App(app_id=app_id, jobs=jobs, deadline=None)

            app_list[app.app_id] = app



            event = Event(event_id=app_id, event_time=submit_time, event_type=Event.APP_SUB, app_id=app_id)
            
            event_queue.append(event)

            sleep_time = int(max(0.01, inter_arrival_times[app_id]))

            fp.write("\n")

            submit_time += timedelta(seconds=sleep_time)

    print("%d Apps generated" % (app_id+1))
    event_queue.reverse()


def run_sim(args):

    models = Models(args.models)

    app_list = {}
    event_queue = list()

    if args.scheduling_policy in ["MCS", "PMCS", "MCS_PRIO"]:



        if args.MCS_config_file == None:
            class_detail = {"num_classes": 1, "class_thresholds": [float('inf')], "class_rates": [frac(1,1)],
                            "clip_demand_factor": 0.01, "delta": 0.1}

        else:
            with open(args.MCS_config_file, "rb") as fp:
                class_detail = pickle.load(fp)


        for key in class_detail:
            print(f"{key}: {class_detail[key]}")


        if args.scheduling_policy == "MCS":
            scheduler = AppMCScheduler(total_gpus=args.num_gpus,
                                        event_queue=event_queue,
                                        app_list=app_list,
                                        class_detail=class_detail,
                                        app_info_fn=args.output_file)
        elif args.scheduling_policy == "PMCS":
            scheduler = AppPracticalMCScheduler(total_gpus=args.num_gpus,
                                        event_queue=event_queue,
                                        app_list=app_list,
                                        class_detail=class_detail,
                                        quantum=100,
                                        app_info_fn=args.output_file)

    elif args.scheduling_policy == "FIFO":
        scheduler = AppPrioScheduler(total_gpus=args.num_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    prio_func=lambda a: a.submit_time,
                                    app_info_fn=args.output_file)
    elif args.scheduling_policy == "SRTF":
        scheduler = AppPrioScheduler(total_gpus=args.num_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    prio_func=lambda a: a.estimated_remaining_service/a.jobs[0].thrpt(a.demand),
                                    app_info_fn=args.output_file)
    elif args.scheduling_policy == "SRSF":
        scheduler = AppPrioScheduler(total_gpus=args.num_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    prio_func=lambda a: a.demand * a.estimated_remaining_service/a.jobs[0].thrpt(a.demand),
                                    app_info_fn=args.output_file)
    elif args.scheduling_policy == "LAS":
        scheduler = AppPrioScheduler(total_gpus=args.num_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    prio_func=lambda a: a.service - a.remaining_service,
                                    app_info_fn=args.output_file)

    elif args.scheduling_policy == "FS":
        scheduler = AppFairScheduler(total_gpus=args.num_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    app_info_fn=args.output_file)
    elif args.scheduling_policy == "PFS":
        scheduler = AppPracticalFairScheduler(total_gpus=args.num_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    quantum=1,
                                    app_info_fn=args.output_file)

    elif args.scheduling_policy == "THEMIS":
        scheduler = AppThemisScheduler(total_gpus=args.num_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    app_info_fn=args.output_file)
    elif args.scheduling_policy == "AFS":
        scheduler = AppAFSScheduler(total_gpus=args.num_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    app_info_fn=args.output_file)
    else:
        raise NotImplementedError


    scheduler._p_error = args.p_error

    if args.estimate:
        scheduler.set_estimator()




    if args.trace:
        gen_workload_from_trace(args.trace, app_list, event_queue, models, max_apps=args.num_apps)
    else:
        gen_workload(args.workload,
                    args.workload,
                    args.workload,
                    args.workload,
                    args.load,
                    args.num_gpus,
                    args.num_apps,
                    args.seed,
                    app_list,
                    event_queue,
                    models)
    
    


    return

    print("Starting sim with %d Apps" % len(event_queue))

    tick = datetime.now()
    
    scheduler.run()
    tock = datetime.now()

    print(f"\nsim took {(tock - tick).total_seconds()} secs")

    if scheduler._num_finished_apps != len(app_list):
        print("Cluster is not big enough for largest job")
        sys.exit(1)

    print("\nSim ended.")


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-trace', help="trace name", choices = TRACES, type=str)
    group.add_argument('-workload', help = "workload name", choices=WORKLOADS, type=str)

    parser.add_argument('-models', choices=["realistic", "toy", "linear"], default="linear", help= "which model type to use")

    parser.add_argument('-load', help = "load", type=float, default=0.8)
    parser.add_argument('-num_gpus', help='num_gpus', default=1, type=int)
    parser.add_argument('-num_apps', help="number of apps to generate", type=int, default=100)

    parser.add_argument('-scheduling_policy', help="Scheduling policy", type=str, default="FIFO")
    parser.add_argument('-logging', help="logging verbosity (0-2)", default=1, type=int)
    parser.add_argument('-estimate', help='whether to estimate ACTs 0/1', default=1, type=int, choices=[0,1])
    parser.add_argument('-output_file', default="sim_result.csv", type=str)
    parser.add_argument('-seed', type=int, default=4567)
    parser.add_argument('-p_error', type=float, default=None)

    parser.add_argument('-MCS_config_file', default=None, type=str)

    args = parser.parse_args()


    run_sim(args)

    


