import ray
from ray import tune
import os, sys, csv

sys.path.append(f"{os.path.expanduser('~')}/automl-setup/schedulers")


import numpy as np
import argparse
import matplotlib.pyplot as plt

import pickle

# from ray.tune.schedulers.hyperband import HyperBandScheduler as HB
# from ray.tune.schedulers.sync_successive_halving import SyncSuccessiveHalving as SHA
from ray.tune.integration.keras import TuneReportCallback
from filelock import FileLock
from ray.util.queue import Queue
from ray.tune.callback import Callback

from datetime import datetime, timedelta
from time import sleep
import logging


from schedulers.common import App, Job, Event
from schedulers.helpers import gen_data_from_cdf
from RayPrioScheduler import RayAppPrioScheduler
from RayMCScheduler import RayAppMCScheduler
from RayFairScheduler import RayAppFairScheduler
from RayAFSScheduler import RayAppAFSScheduler
from RayThemisScheduler import RayAppThemisScheduler

from fractions import Fraction as frac
    


@ray.remote
def training_job(app, event_queue, inactivity_time):

    from ray.air import session, Checkpoint
    from ray.air.config import ScalingConfig
    from ray.air.config import RunConfig
    from ray.train.tensorflow import TensorflowTrainer
    import tensorflow_datasets as tfds
    from ray.tune import Stopper



    def concurrency_manager(queue, block=False):

        latest_allocation = None


        if queue != None:

            if block:
                latest_allocation = queue.get()

            num_gets = queue.size()

            for _ in range(num_gets):
                latest_allocation = queue.get()
        return latest_allocation


    class ResourceChangeStopper(Stopper):
        """docstring for ResourceChangeStopper"""
        def __init__(self, allocation, queue):
            super(ResourceChangeStopper, self).__init__()
            self._allocation = allocation
            self._queue = queue

        def __call__(self, trial_id, result):

            latest_allocation = concurrency_manager(self._queue)

            if latest_allocation != None and latest_allocation != self._allocation:
                self._queue.put(latest_allocation)
                return True
            return False

        def stop_all(self):
            return False

            




    class CustomCallBack(ray.air.integrations.keras.ReportCheckpointCallback):
        """docstring for CustomCallBack"""
        def __init__(self, app):
            super(CustomCallBack, self).__init__()

            self.uplink_queue = app.trial_runner_queue["uplink"]
            self.app_id = app.app_id
            self.epochs = app.config["epochs"]

        def on_epoch_begin(self, epoch, logs):
            if session.get_world_rank() == 0:
                pass
                # self.event_queue.put(Event(event_id=self.app_id, event_type=Event.APP_PING, event_time=datetime.now(), app_id=self.app_id, job_id=self.app_id))
            
        def on_epoch_end(self, epoch, logs):
            
            if session.get_world_rank() == 0:
                self.uplink_queue.put({"trial_0": self.epochs - epoch})


    def create_dataset(dname):
        # Download the CIFAR-10 dataset.

        train_d, test_d = tfds.load(dname, split=["train", "test"])

        train_d = ray.data.from_tf(train_d)
        test_d = ray.data.from_tf(test_d)

        return train_d, test_d









    def train_loop_per_worker(app):



        import tensorflow as tf
        from model_gen import model_generator
        from ray.train.tensorflow import TensorflowCheckpoint
        from ray.train.data_parallel_trainer import _load_checkpoint_dict
        
        # set_memory_growth_on_worker()



        dataset_shard = session.get_dataset_shard("train")

        options = tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
        )


        config = app.config


        strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=options)
        # strategy = tf.distribute.MultiWorkerMirroredStrategy()
        with strategy.scope():

            model = model_generator({"p1": config["model"]["p1"],
                                     "p2": config["model"]["p2"],
                                     "p3": config["model"]["p3"],
                                     "p4": config["model"]["p4"],
                                     "p5": config["model"]["p5"]})

            if app.ckpt != None:
                
                model_weights, _ = _load_checkpoint_dict(app.ckpt, "TensorflowTrainer")
                model.set_weights(model_weights)

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            loss = tf.keras.losses.SparseCategoricalCrossentropy()

            model.compile(loss=loss,
                    optimizer=optimizer,
                    metrics=['accuracy'])


        batch_size = config["batch_size"]//session.get_world_size()

        tf_dataset = dataset_shard.to_tf(
            feature_columns="image",
            label_columns="label",
            batch_size=batch_size
        )

        epochs = config["epochs"]

        for i in range(epochs):
            model.fit(tf_dataset, batch_size = config["batch_size"])
            session.report(metrics={"remaining_epochs": epochs - (i+1)}, checkpoint=TensorflowCheckpoint.from_model(model))
        return




    train_dataset, test_dataset = create_dataset("cifar10")

    app.ckpt = None

    done = False


    while not done:

        allocation = concurrency_manager(app.trial_runner_queue["downlink"], block=True)

        if allocation != None and allocation != 0:
            trainer = TensorflowTrainer(
                train_loop_per_worker=train_loop_per_worker,
                scaling_config=ScalingConfig(num_workers=allocation, use_gpu=True),
                run_config=RunConfig(stop=[ResourceChangeStopper(allocation, app.trial_runner_queue["downlink"])]),
                datasets={"train": train_dataset},
                train_loop_config=app)

            if app.ckpt == None:
                event_queue.put(Event(event_id=app.app_id, event_type=Event.APP_START, event_time=datetime.now(), app_id=app.app_id, job_id=app.app_id))
                event_queue.put(Event(event_id=app.app_id, event_type=Event.JOB_START, event_time=datetime.now(), app_id=app.app_id, job_id=0))
                
            result = trainer.fit()
            app.ckpt = result.checkpoint
            app.config["epochs"] = result.metrics["remaining_epochs"]

            if result.metrics["remaining_epochs"] == 0:
                done = True

    '''
    ckpt = result.checkpoint
    metrics = result.metrics

    app.ckpt = result.checkpoint
    app.config["epochs"] = metrics["remaining_epochs"]

    trainer = TensorflowTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=ScalingConfig(num_workers=app.num_workers//2, use_gpu=True),
        datasets={"train": train_dataset},
        train_loop_config=app)

    result = trainer.fit()
    '''
    event_queue.put(Event(event_id=app.app_id, event_type=Event.JOB_END, event_time=datetime.now(), app_id=app.app_id, job_id=0))
    event_queue.put(Event(event_id=app.app_id, event_type=Event.APP_END, event_time=datetime.now(), app_id=app.app_id, job_id=0))

@ray.remote
def app_generator(app_list, event_queue):

    start_time = datetime.now()
    submit_time = datetime.now()
        
    for app_id in app_list:

        app = app_list[app_id]

        sleep(app.sleep_time)

        event = Event(event_id=app_id, event_time=datetime.now(), event_type=Event.APP_SUB, app_id=app_id)
        event_queue.put(event)


def gen_workload_from_trace(fname, app_list, event_queue):
    # app trace should follow app_id,total_stages,submit_time,job_id,num_gpu,_,stage_id,_,duration,deadline format
    # app list is a dictionary mapping from app_id to object App
    

    with open(fname, 'r') as fp:
        csvReader = csv.reader(fp)
        next(csvReader)
        for row in csvReader:

            app_id,submit_time,service,num_jobs,sleep_time = row

            app_id = int(app_id)
            service = float(service)
            num_jobs = int(num_jobs)
            sleep_time = float(sleep_time)

            jobs = {}

            for job_id in range(num_jobs):

                jobs[job_id] = Job(app_id=app_id, job_id=job_id, service = service/num_jobs,
                                    demand=1,
                                    min_demand=1)

                jobs[job_id].thrpt_dic = [0,1.0]


            '''
            job_id = 0
            jobs[job_id] = Job(app_id=app_id, job_id=job_id, service=service,
                                demand=num_jobs,
                                min_demand=np.random.choice(min_gpus_per_job))
            '''


            app = App(app_id=app_id, jobs=jobs, deadline=None)
            app.exec_func = tune_cifar10
            app.sleep_time = sleep_time
            app_list[app.app_id] = app


            print("\r%d Apps generated" % (app_id+1),end='')
    print("")


    app_generator.remote(app_list, event_queue)


def gen_workload(cdf_app_service_times, cdf_num_jobs_per_app, cdf_max_gpus_per_job, cdf_min_gpus_per_job, load, num_gpus, num_apps, seed, app_list, event_queue):


    np.random.seed(seed)


    file_dir = os.path.dirname(os.path.abspath(__file__))



    app_service_times = gen_data_from_cdf(f"{file_dir}/schedulers/cdfs/cdf-app-service-times-{cdf_app_service_times}.csv",
                                        num_points=num_apps, dtype=int, interpolation=True)
    num_jobs_per_app = gen_data_from_cdf(f"{file_dir}/schedulers/cdfs/cdf-num-jobs-per-app-{cdf_num_jobs_per_app}.csv",
                                        num_points=num_apps, dtype=int, interpolation=True)
    max_gpus_per_job = gen_data_from_cdf(f"{file_dir}/schedulers/cdfs/cdf-max-gpus-per-job-{cdf_max_gpus_per_job}.csv",
                                        num_points=100, dtype=int, interpolation=True)
    min_gpus_per_job = gen_data_from_cdf(f"{file_dir}/schedulers/cdfs/cdf-min-gpus-per-job-{cdf_min_gpus_per_job}.csv",
                                        num_points=100, dtype=int, interpolation=True)
    
    avg_interarrival_time = (np.mean(app_service_times))/((load)*num_gpus)
    sleep_times = [0.0] + list(map(lambda s: int(max(1,s)), np.random.exponential(avg_interarrival_time, num_apps-1)))
    

    submit_time = 0

    with open(f"{file_dir}/workload.csv",'w') as fp:
        fp.write("app_id,submit_time,service,num_jobs,sleep_time\n")
        
        for app_id in range(num_apps):

            num_jobs = num_jobs_per_app[app_id]
            service = max(int(float(app_service_times[app_id])/num_jobs), 30) * num_jobs

            jobs = {}

            for job_id in range(num_jobs):
                jobs[job_id] = Job(app_id=app_id, job_id=job_id, service = (service/num_jobs),
                                    demand=np.random.choice(max_gpus_per_job),
                                    min_demand=np.random.choice(min_gpus_per_job))

        
            app = App(app_id=app_id, jobs=jobs, deadline=None)
            app.exec_func = tune_cifar10            
            app.sleep_time = sleep_times[app_id]
            app_list[app.app_id] = app


            submit_time += app.sleep_time
            fp.write(f"{app.app_id},{submit_time},{app.remaining_service},{len(app.jobs)},{app.sleep_time}\n")


            print("\r%d Apps generated" % (app_id+1),end='')

    print("")

    app_generator.remote(app_list, event_queue)

 


if __name__ == '__main__':

    app_list = {}
    event_queue = Queue()

    total_gpus = 4
    output_file = "100job_run_test.csv"

    scheduler = RayAppPrioScheduler(total_gpus=total_gpus,
                                event_queue=event_queue,
                                app_list=app_list,
                                prio_func=lambda a: a.submit_time,
                                app_info_fn=output_file)




    def create_app(app_id, num_workers, sleep_time):
        job_id = 0

        batch_size = 2048
        epochs = 10
        p1,p2,p3,p4,p5 = 1,1,1,1,1

        
        jobs = {}

        jobs[job_id] = Job(app_id=app_id, job_id=job_id, service = 100, demand= num_workers, min_demand=1)


        jobs[job_id].thrpt_dic = [0,1.0,2.0,3.0,4.0]


        app = App(app_id=app_id, jobs=jobs, deadline=None)
        app.exec_func = training_job            
        app.sleep_time = sleep_time

        app.num_workers = num_workers
        app.config = {"epochs": epochs, "batch_size": batch_size, "verbosity": 0,
                    "model": {"p1": int(p1), "p2": int(p2), "p3": int(p3), "p4": int(p4), "p5": int(p5)}}

        return app

    sleep_time = 0
    avg_interarrival_time = 150
    for app_id in range(100):
        num_gpus = np.random.choice([1,2,4])
        app = create_app(app_id, num_gpus, sleep_time)
        app_list[app.app_id] = app
        sleep_time = np.random.exponential(avg_interarrival_time)


    submit_time=0
    print("\r%d Apps generated" % (len(app_list)),end='')
    print(r"\n")
    app_generator.remote(app_list, event_queue)



    tick = datetime.now()
    scheduler.run()
    tock = datetime.now()
    

    sleep(10.0)


    if scheduler._num_finished_apps != len(app_list):
        print("Cluster is not big enough for largest job")
        sys.exit(1)


    print("\nExpt ended.")



'''
if __name__ == '__main__':



    parser = argparse.ArgumentParser()

    parser.add_argument('-head_ip', help="IP address of head ray node", type=str, default="10.1.1.2")
    parser.add_argument('-head_port', help="port# of head ray node", type=str, default="6379")
    parser.add_argument('-from_trace', help="1/0 to generate workload using trace", type=int, default=0)
    parser.add_argument('-cdf_app_service_times', help = "fname of app service times", type=str, default="small")
    parser.add_argument('-cdf_num_jobs_per_app', help = "fname of num jobs per app", type=str, default="small")
    parser.add_argument('-cdf_max_gpus_per_job', help = "fname of max gpus per job", type=str, default="1GPU")
    parser.add_argument('-cdf_min_gpus_per_job', help = "fname of min gpus per job", type=str, default="0GPU")
    parser.add_argument('-num_apps', help="number of apps to generate", type=int, default=1)

    parser.add_argument('-load', help = "load", type=float, default=0.8)


    parser.add_argument('-scheduling_policy', help="Scheduling policy", type=str, default="MCS")
    parser.add_argument('-logging', help="logging verbosity (0-2)", default=1, type=int)
    parser.add_argument('-estimation_policy', help='estimation_policy', default='MAX', type=str)
    parser.add_argument('-output_file', default="results.csv", type=str)
    parser.add_argument('-seed', type=int, default=4567)

    parser.add_argument('-MCS_config_file', default=None, type=str)

    args = parser.parse_args()


    ray.init(address=f"{args.head_ip}:{args.head_port}", _redis_password="tf_cluster_123")
    total_gpus = ray.cluster_resources()["GPU"]
    scheduling_policy = args.scheduling_policy
    output_file = args.output_file



    app_list = {}
    event_queue = Queue()



    if scheduling_policy == "MCS":


        # class_detail = {"num_classes": 2, "class_thresholds": [500.0, float('inf')], "class_rates": [0.75,0.25]}

        if args.MCS_config_file == None:
            class_detail = {"num_classes": 3, "class_thresholds": [1523, 5088, float('inf')], "class_rates": [frac(889,1000),frac(1,10),frac(11,1000)]}
        else:
            with open(args.MCS_config_file, "rb") as fp:
                class_detail = pickle.load(fp)


        print(class_detail["num_classes"])
        print(class_detail["class_thresholds"])
        print(class_detail["class_rates"])


        scheduler = RayAppMCScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    class_detail=class_detail,
                                    quantum=200,
                                    app_info_fn=output_file)
    elif scheduling_policy == "SRTF":


        scheduler = RayAppPrioScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    prio_func=lambda a: a.remaining_service/a.demand if a.demand > 0 else a.remaining_service,
                                    app_info_fn=output_file)

    elif scheduling_policy == "SRSF":

        scheduler = RayAppPrioScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    prio_func=lambda a: a.remaining_service,
                                    app_info_fn=output_file)

    elif scheduling_policy == "LAS":

        scheduler = RayAppPrioScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    prio_func=lambda a: a.service - a.remaining_service,
                                    app_info_fn=output_file)

    elif scheduling_policy == "FIFO":

        scheduler = RayAppPrioScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    prio_func=lambda a: a.submit_time,
                                    app_info_fn=output_file)

    elif scheduling_policy == "FS":
        scheduler = RayAppFairScheduler(total_gpus=total_gpus,
                                                event_queue=event_queue,
                                                app_list = app_list,
                                                quantum=200,
                                                app_info_fn=output_file)

    elif scheduling_policy == "AFS":
        scheduler = RayAppAFSScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    app_info_fn=output_file)

    elif scheduling_policy == "THEMIS":
        scheduler = RayAppThemisScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    quantum=200,
                                    app_info_fn=output_file)
    else:
        raise NotImplementedError



    if args.from_trace:
        gen_workload_from_trace("workload.csv", app_list, event_queue)
    else:
        gen_workload(args.cdf_app_service_times,
                    args.cdf_num_jobs_per_app,
                    args.cdf_max_gpus_per_job,
                    args.cdf_min_gpus_per_job,
                    args.load,
                    total_gpus,
                    args.num_apps,
                    args.seed,
                    app_list,
                    event_queue)


    print("Starting experiment with %d Apps" % len(app_list))

    
    tick = datetime.now()
    scheduler.run()
    tock = datetime.now()
    

    sleep(10.0)


    if scheduler._num_finished_apps != len(app_list):
        print("Cluster is not big enough for largest job")
        sys.exit(1)


    print("\nExpt ended.")
'''
pass