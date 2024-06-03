import ray
from ray import tune
import os, sys, csv

import numpy as np
import argparse
import matplotlib.pyplot as plt

import pickle


# implemented
from ray.tune.schedulers.timed_fifo import TimedFIFOScheduler as TrialScheduler


from ray.tune.integration.keras import TuneReportCallback

from tensorflow.keras import datasets, layers, models, regularizers, Input

from filelock import FileLock
from ray.util.queue import Queue


from datetime import datetime, timedelta
from time import sleep
import logging


from common import App, Job, Event

from RayPrioScheduler import RayAppPrioScheduler
from RayMCScheduler import RayAppMCScheduler
from RayFairScheduler import RayAppFairScheduler
from RayAFSScheduler import RayAppAFSScheduler
from RayThemisScheduler import RayAppThemisScheduler

from fractions import Fraction as frac
from my_trial_executor import MyRayTrialExecutor
from ray.tune.resources import Resources

def model_generator(config):
    
    weight_decay = 0.0005


    dropout_scale = 1


    inputs = Input(shape=(32,32,3))

    l = layers.Conv2D(64, (3, 3), padding='same', activation= 'relu',kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    l = layers.BatchNormalization()(l)
    l = layers.Dropout(0.3 * (dropout_scale))(l)

    if config['p1'] == 1:
        l = layers.Conv2D(64, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
        l = layers.BatchNormalization()(l)
    
    l = layers.MaxPooling2D(pool_size=(2, 2))(l)


    ###################################################################


    l = layers.Conv2D(128, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
    l = layers.BatchNormalization()(l)
    l = layers.Dropout(0.4 * (dropout_scale))(l)

    if config['p2'] == 1:
        l = layers.Conv2D(128, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
        l = layers.BatchNormalization()(l)
    
    l = layers.MaxPooling2D(pool_size=(2, 2))(l)


    ###################################################################


    l = layers.Conv2D(256, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
    l = layers.BatchNormalization()(l)
    l = layers.Dropout(0.4 * (dropout_scale))(l)

    l = layers.Conv2D(256, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
    l = layers.BatchNormalization()(l)
    l = layers.Dropout(0.4 * (dropout_scale))(l)

    if config['p3'] == 1:
        l = layers.Conv2D(256, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
        l = layers.BatchNormalization()(l)
    
    l = layers.MaxPooling2D(pool_size=(2, 2))(l)


    ###################################################################


    l = layers.Conv2D(512, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
    l = layers.BatchNormalization()(l)
    l = layers.Dropout(0.4 * (dropout_scale))(l)

    l = layers.Conv2D(512, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
    l = layers.BatchNormalization()(l)
    l = layers.Dropout(0.4 * (dropout_scale))(l)

    if config['p4'] == 1:
        l = layers.Conv2D(512, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
        l = layers.BatchNormalization()(l)
        
    l = layers.MaxPooling2D(pool_size=(2, 2))(l)


    ###################################################################


    l = layers.Conv2D(512, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
    l = layers.BatchNormalization()(l)
    l = layers.Dropout(0.4 * (dropout_scale))(l)

    l = layers.Conv2D(512, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
    l = layers.BatchNormalization()(l)
    l = layers.Dropout(0.4 * (dropout_scale))(l)

    if config['p5'] == 1:
        l = layers.Conv2D(512, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
        l = layers.BatchNormalization()(l)
    
    l = layers.MaxPooling2D(pool_size=(2, 2))(l)


    ###################################################################

    l = layers.Flatten()(l)
    output = layers.Dense(10, activation='softmax')(l)

    model = models.Model(inputs=inputs,outputs=output)

    return model


def train_cifar10(config, checkpoint_dir=None):

    import tensorflow as tf

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    logging.getLogger('tensorflow').setLevel(logging.FATAL)


    batch_size = 128
    epochs = 10000
    num_classes = 10

    N_train = 35000
    N_valid = 15000


    with FileLock(os.path.expanduser("~/.data.lock")):
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()


    N = (x_train.shape)[0]

    assert(N_train+N_valid == N)

    '''
    x_train = x_train[:5000]
    y_train = y_train[:5000]

    N_train = 4800
    N_valid = 200
    '''


    def normalize(X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test


    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = normalize(x_train, x_test)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)


    x_train, y_train = x_train[:N_train], y_train[:N_train]
    x_validation, y_validation = x_train[-N_valid:], y_train[-N_valid:]
    x_test, y_test = x_test, y_test


    ###################################################################

    model = model_generator(config)

    if checkpoint_dir:
        model = tf.keras.models.load_model(os.path.join(checkpoint_dir, "model.keras"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.get('lr', 0.001))
    loss = tf.keras.losses.CategoricalCrossentropy()

    model.compile(loss=loss,
                optimizer=optimizer,
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(x_validation, y_validation),
            callbacks=[TuneReportCallback({"accuracy": "accuracy"})])

@ray.remote
def tune_cifar10(app, event_queue, inactivity_time):

    trial_scheduler = TrialScheduler(time_attr='time_total_s',budget=(app.service/app.demand))


    get_queue = app.trial_runner_queue.get('downlink')
    set_queue = app.trial_runner_queue.get('uplink')


    # TODO: set os.environ["TUNE_CLUSTER_SSH_KEY"] = f"{os.path.expanduser('~')}/.ssh/key"
    trial_executor = MyRayTrialExecutor(
                        name=f"app_{app.app_id}",
                        get_queue=get_queue,
                        set_queue=set_queue,
                        event_queue=event_queue,
                        init_resources = Resources(cpu=app.allocation,gpu=app.allocation),
                        inactivity_time=inactivity_time,
                    )

    analysis = tune.run(
        train_cifar10,
        resources_per_trial={'cpu': 1, 'gpu': 1},
        name=f"app_{app.app_id}",
        trial_name_creator=lambda T: "app_%d_%s" % (app.app_id, T.trial_id),
        scheduler=trial_scheduler,
        num_samples=app.demand,
        config={"p1": tune.choice([0]),
                "p2": tune.choice([0]),
                "p3": tune.choice([0]),
                "p4": tune.choice([0]),
                "p5": tune.choice([0]),
                "lr": tune.choice([0.1,0.01,0.0001])},
        trial_executor=trial_executor,
        verbose=1,
    )


@ray.remote
def app_generator(app_list, event_queue):

    start_time = datetime.now()
    submit_time = datetime.now()
        
    for app_id in app_list:

        app = app_list[app_id]

        sleep(app.sleep_time)

        event = Event(event_id=app_id, event_time=datetime.now(), event_type=Event.APP_SUB, app_id=app_id)
        event_queue.put(event)


def gen_workload_from_trace(trace_name, app_list, event_queue):
    # app trace should follow app_id,total_stages,submit_time,job_id,num_gpu,_,stage_id,_,duration,deadline format
    # app list is a dictionary mapping from app_id to object App
    
    file_dir = os.path.dirname(os.path.abspath(__file__))
    

    with open(f"{file_dir}/traces/{trace_name}.csv", 'r') as fp:
        csvReader = csv.reader(fp)
        next(csvReader)
        for row in csvReader:


            if "#" in "".join(row):
                continue

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


            app = App(app_id=app_id, jobs=jobs, deadline=None)
            app.exec_func = tune_cifar10
            app.sleep_time = sleep_time
            app_list[app.app_id] = app


            print("\r%d Apps generated" % (app_id+1),end='')
    print("")


    app_generator.remote(app_list, event_queue)
    
if __name__ == '__main__':



    parser = argparse.ArgumentParser()

    parser.add_argument('-trace', help="trace name", type=str, default="toy_trace")
    parser.add_argument('-scheduling_policy', help="Scheduling policy", type=str, default="MCS")
    parser.add_argument('-logging', help="logging verbosity (0-2)", default=1, type=int)
    parser.add_argument('-output_file', default="results.csv", type=str)
    parser.add_argument('-seed', type=int, default=4567)
    parser.add_argument('-MCS_config_file', default=None, type=str)

    args = parser.parse_args()


    ray.init(address="auto")
    total_gpus = ray.cluster_resources()["GPU"]
    scheduling_policy = args.scheduling_policy
    output_file = args.output_file



    app_list = {}
    event_queue = Queue()


    if scheduling_policy == "MCS":


        # class_detail = {"num_classes": 2, "class_thresholds": [500.0, float('inf')], "class_rates": [0.75,0.25]}

        if args.MCS_config_file == None:
            class_detail = {"num_classes": 3, "class_thresholds": [1523, 5088, float('inf')], "class_rates": [frac(889,1000),frac(1,10),frac(11,1000)], "clip_demand_factor": 0.01, "delta": 0.01}
        else:
            with open(args.MCS_config_file, "rb") as fp:
                class_detail = pickle.load(fp)


        print(class_detail["num_classes"])
        print(class_detail["class_thresholds"])
        print(class_detail["class_rates"])
        print(class_detail["clip_demand_factor"])
        print(class_detail["delta"])


        scheduler = RayAppMCScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    class_detail=class_detail,
                                    quantum=600,
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
                                                quantum=600,
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
                                    quantum=600,
                                    app_info_fn=output_file)
    else:
        raise NotImplementedError




    gen_workload_from_trace(args.trace, app_list, event_queue)


    print("Starting experiment with %d Apps" % len(app_list))

    
    tick = datetime.now()
    scheduler.run()
    tock = datetime.now()
    

    sleep(10.0)


    if scheduler._num_finished_apps != len(app_list):
        print("Cluster is not big enough for largest job")
        sys.exit(1)


    print("\nExpt ended.")
