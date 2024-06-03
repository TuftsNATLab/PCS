from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
    Any,
)

import ray
from ray import tune
import numpy as np
import argparse
from functools import partial


from ray.tune.schedulers.timed_fifo import TimedFIFOScheduler as TrialScheduler
from ray.tune.utils.placement_groups import PlacementGroupFactory


from ray.tune.trial import Trial, Checkpoint, Location, TrialInfo
from ray.tune.schedulers import ResourceChangingScheduler, ASHAScheduler
from ray.tune import Trainable
from ray.tune.resources import Resources

from tensorflow.keras import datasets, layers, models, regularizers, Input
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.integration.keras import TuneReportCheckpointCallback


from ray.tune.ray_trial_executor import *

from ray.tune.result import TRIAL_INFO, STDOUT_FILE, STDERR_FILE

from filelock import FileLock
import os


from ray.util.queue import Queue, Empty

from common import Event

import time


import copy


from my_trial_executor import MyRayTrialExecutor


CHECKPOINT_FILENAME="my_model.keras"


class App(object):
    """docstring for App"""
    def __init__(self, app_id, service, demand, trial_runner_queue, allocation):
        super(App, self).__init__()
        self.app_id = app_id
        self.service = service
        self.demand = demand
        self.trial_runner_queue = trial_runner_queue
        self.allocation = allocation


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



def train_cifar10(config: dict, checkpoint_dir=None):

    import tensorflow as tf


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

    # load weights
    if checkpoint_dir:
        model = tf.keras.models.load_model(os.path.join(checkpoint_dir, CHECKPOINT_FILENAME))


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
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

    # model.fit(x_train, y_train,
    #         batch_size=batch_size,
    #         epochs=epochs,
    #         verbose=0,
    #         validation_data=(x_validation, y_validation),
    #         callbacks=[
    #             TuneReportCheckpointCallback(
    #                 filename=CHECKPOINT_FILENAME,
    #                 # checkpointing should happen every iteration
    #                 # with dynamic resource allocation
    #                 frequency=1)


    #         ])


@ray.remote
def schedule_q_put(sleep_time, queue, item):
    time.sleep(sleep_time)
    queue.put(item)


def example_resources_allocation_function(
        trial_runner: "trial_runner.TrialRunner", trial: Trial,
        result: Dict[str, Any], scheduler: "ResourceChangingScheduler"
) -> Union[None, PlacementGroupFactory, Resources]:
    base_trial_resource = scheduler._base_trial_resources or PlacementGroupFactory([{"CPU": 1, "GPU": 1}])
    min_gpu = base_trial_resource.required_resources.get("GPU", 1)

    # Get the number of GPUs available in total (not just free)
    total_available_gpus = (trial_runner.trial_executor._avail_resources.gpu)

    get_trial_idx = lambda t: int(t.trial_id.split('_')[-1])


    all_trials = trial_runner.get_live_trials()
    trial_idxs = [get_trial_idx(t) for t in all_trials]

    gpu_allocations = [0] * len(all_trials)

    per_trial_gpu_allocations = 1

    print(f"DEBUG: {trial_idxs}")

    for trial_idx in trial_idxs:
        allocation = min(per_trial_gpu_allocations, total_available_gpus)
        gpu_allocations[trial_idx] = allocation
        total_available_gpus -= allocation


    gpu_allocation = gpu_allocations[get_trial_idx(trial)]

    # trial_runner.trial_executor.resources
    print("++++++++++++++++++++++")
    print(f"Trial_id: {trial.trial_id} has resources: {trial.resources} and placement group: {trial.placement_group_factory} and total gpus are: {trial_runner.trial_executor._avail_resources.gpu} and has been assigned {gpu_allocation}")
    print("++++++++++++++++++++++")


    # # Divide the free CPUs among all live trials
    # cpu_to_use = max(
    #     min_cpu,
    #     total_available_cpus // len(trial_runner.get_live_trials()))
    # Assign new CPUs to the trial in a PlacementGroupFactory



    return PlacementGroupFactory([{"CPU": 1 if gpu_allocation > 0 else 0, "GPU": gpu_allocation}])
    # return PlacementGroupFactory([{"CPU": 1, "GPU": total_available_gpus//len(trial_runner.get_live_trials())}])



@ray.remote
def tune_cifar10(app, event_queue, inactivity_time, verbose=3):

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
        verbose=verbose,
    )


'''
def tune_cifar10(num_samples=2, reduction_factor=2, budget=10.0, sleep_time=None):

    app = App(0, budget, num_samples)

    trial_scheduler=TrialScheduler(time_attr='time_total_s',budget=(app.service/app.demand))

    queue1 = Queue()
    queue2 = Queue()
    queue3 = Queue()
    
    if sleep_time:
        schedule_q_put.remote(sleep_time, queue1, Resources(cpu=1,gpu=1))

    trial_executor = MyRayTrialExecutor(
                        name=f"app_{app.app_id}",
                        get_queue=queue1,
                        set_queue=queue2,
                        event_queue=queue3,
                        init_resources = Resources(cpu=2,gpu=2),
                    )


    analysis = tune.run(
        train_cifar10,
        # resources_per_trial=PlacementGroupFactory([{"CPU": 1, "GPU": 1}]),
        resources_per_trial={'cpu': 1, 'gpu': 1},
        name=f"app_{app.app_id}",
        trial_name_creator=lambda T: "app_%d_%s" % (app.app_id, T.trial_id),
        scheduler=trial_scheduler,
        num_samples=num_samples,
        config={"p1": tune.choice([0]),
                "p2": tune.choice([0]),
                "p3": tune.choice([0]),
                "p4": tune.choice([0]),
                "p5": tune.choice([0])},
        
        trial_executor=trial_executor,
        verbose=1,
    )
        


    print("Best config: ", analysis.get_best_config(
        metric="accuracy", mode="max"))


    print("===================")
    while not queue3.empty():
        event = queue3.get()
        print(f"Event recieved: {event}")
        
'''


def multi_app_test(args):
    
    ray.init(address="auto")

    assert(args.allocation <= ray.available_resources()['GPU'])

    event_queue = Queue()

    app0 = App(app_id=0, service=120, demand=4, trial_runner_queue={"downlink": Queue(), "uplink": Queue()}, allocation=4)
    app1 = App(app_id=1, service=120, demand=4, trial_runner_queue={"downlink": Queue(), "uplink": Queue()}, allocation=4)

    
    app0.trial_runner_queue['downlink'].put(4)
    future0 = tune_cifar10.remote(app0, event_queue, inactivity_time=None, verbose=1)
    app0.trial_runner_queue['downlink'].put(0)

    app1.trial_runner_queue['downlink'].put(4)
    future1 = tune_cifar10.remote(app1, event_queue, inactivity_time=None, verbose=1)
    ray.get([future1])
    app0.trial_runner_queue['downlink'].put(4)
    ray.get([future0])

    time.sleep(2)




def single_app_test(args):
    ray.init(address="auto")

    assert(args.allocation <= ray.available_resources()['GPU'])

    event_queue = Queue()

    app0 = App(app_id=0, service=120, demand=4, trial_runner_queue={"downlink": Queue(), "uplink": Queue()}, allocation=4)
    # app1 = App(app_id=1, service=120, demand=4, trial_runner_queue={"downlink": Queue(), "uplink": Queue()}, allocation=4)

    
    app0.trial_runner_queue['downlink'].put(4)
    future0 = tune_cifar10.remote(app0, event_queue, inactivity_time=None, verbose=0)
    # app0.trial_runner_queue['downlink'].put(0)

    # app1.trial_runner_queue['downlink'].put(4)
    # future1 = tune_cifar10.remote(app1, event_queue, inactivity_time=None, verbose=0)

    ray.get([future0])
    time.sleep(2)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=14)
    parser.add_argument("--reduction_factor", type=int, default=2)
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--allocation", type=int, default=1)
    parser.add_argument("--sleep_time", type=float, default=None)
    args = parser.parse_args()

    # single_app_test(args)
    multi_app_test(args)

    # os.environ["TUNE_CLUSTER_SSH_KEY"] = f"{os.path.expanduser('~')}/.ssh/key"

