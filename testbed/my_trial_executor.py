# coding: utf-8
import copy
from collections import deque
from functools import partial
import logging
import os
import random
import time
import traceback
from contextlib import contextmanager
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
)

import ray
from ray.actor import ActorHandle
from ray.exceptions import GetTimeoutError
from ray import ray_constants
from ray._private.resource_spec import ResourceSpec, NODE_ID_PREFIX
from ray.tune.durable_trainable import DurableTrainable
from ray.tune.error import AbortTrialExecution, TuneError
from ray.tune.logger import NoopLogger
from ray.tune.result import TRIAL_INFO, STDOUT_FILE, STDERR_FILE
from ray.tune.resources import Resources
from ray.tune.utils.placement_groups import PlacementGroupManager, \
    get_tune_pg_prefix
from ray.tune.utils.trainable import TrainableUtil
from ray.tune.trial import Trial, Checkpoint, Location, TrialInfo
from ray.tune.trial_executor import TrialExecutor
from ray.tune.utils import warn_if_slow
from ray.util import log_once
from ray.util.annotations import DeveloperAPI


from ray.tune.schedulers import TrialScheduler
from ray.tune.trial_executor import TrialExecutor
from ray.util.queue import Queue, Empty

from ray.tune.ray_trial_executor import (
    RayTrialExecutor,
    _ActorClassCache,
    _class_cache,
    _LocalWrapper,
    _TrialCleanup,
    noop_logger_creator,
    _to_gb,
    TUNE_STATE_REFRESH_PERIOD,
    BOTTLENECK_WARN_PERIOD_S,
    NONTRIVIAL_WAIT_TIME_THRESHOLD_S,
    DEFAULT_GET_TIMEOUT,
    TRIAL_CLEANUP_THRESHOLD
)


from common import Event
import datetime

logger = logging.getLogger(__name__)





class MyRayTrialExecutor(RayTrialExecutor):
    """An implementation of MyRayTrialExecutor based on RayTrialExecutor."""

    def __init__(self,
                name,
                get_queue,
                set_queue,
                event_queue,
                init_resources: Resources = Resources(cpu=0,gpu=0),
                inactivity_time: float = float('inf'),
                queue_trials: bool = False,
                reuse_actors: bool = False,
                result_buffer_length: Optional[int] = None,
                refresh_period: Optional[float] = None,
                wait_for_placement_group: Optional[float] = None,
                ):
        
        
        self._name = name
        self._get_queue = get_queue
        self._set_queue = set_queue
        self._event_queue = event_queue

        self._tune_started = False
        self._inactivity_time = inactivity_time or float('inf')
        self._last_ping_time = datetime.datetime.now()

        self._has_errored = False
        self._max_queue_gets = 20

        self._avail_resources = init_resources


        self._max_pending_trials = 10000


        '''
        Everything below it is mostly ray_trial_executor
        '''
        
        TrialExecutor.__init__(self, queue_trials)

        # Check for if we are launching a trial without resources in kick off
        # autoscaler.
        self._trial_queued = False
        self._running = {}
        # Since trial resume after paused should not run
        # trial.train.remote(), thus no more new remote object ref generated.
        # We use self._paused to store paused trials here.
        self._paused = {}

        force_trial_cleanup = int(
            os.environ.get("TUNE_FORCE_TRIAL_CLEANUP_S", "0"))
        self._trial_cleanup = _TrialCleanup(force_cleanup=force_trial_cleanup)
        
        self._reuse_actors = reuse_actors

        self._avail_resources = Resources(cpu=0, gpu=0)
        self._committed_resources = Resources(cpu=0, gpu=0)

        self._staged_trials = set()
        self._just_staged_trials = set()
        self._trial_just_finished = False
        self._trial_just_finished_before = False

        self._resources_initialized = False

        if refresh_period is None:
            refresh_period = float(
                os.environ.get("TUNE_STATE_REFRESH_PERIOD",
                               TUNE_STATE_REFRESH_PERIOD))
        self._refresh_period = refresh_period

        self._default_buffer_length = result_buffer_length or int(
            os.getenv("TUNE_RESULT_BUFFER_LENGTH", 1000))
        self._buffer_length = result_buffer_length

        self._buffer_min_time_s = float(
            os.getenv("TUNE_RESULT_BUFFER_MIN_TIME_S", 0.))
        self._buffer_max_time_s = float(
            os.getenv("TUNE_RESULT_BUFFER_MAX_TIME_S", 100.))

        self._last_resource_refresh = float("-inf")
        self._last_ip_refresh = float("-inf")
        self._last_ip_addresses = set()
        self._last_nontrivial_wait = time.time()

        if ray.is_initialized():
            self._update_avail_resources()

        self._demand = Resources(cpu=0,gpu=0)

        self._steps = 0


    '''
    Everything above is mostly unmodified stuff from ray trial executor
    '''


    def set_max_pending_trials(self, max_pending: int) -> None:
        self._max_pending_trials = max(max_pending, self._max_pending_trials)



    def stage_and_update_status(self, trials: Iterable[Trial]):
        """Check and update statuses of scheduled placement groups.

        Stages placement groups of all trials.
        """
        return True

    def in_staging_grace_period(self) -> bool:
        """Returns True if trials have recently been staged."""
        return False

    def cleanup(self, trials: List[Trial]) -> None:
        self._trial_cleanup.cleanup(partial=False)


    def _get_app_id(self):
        try:
            return int(self._name.split("app_")[-1])
        except Exception as e:
            raise e


    def _get_job_id(self, trial):
        try:
            return int(trial.trial_id.split("_")[-1])
        except Exception as e:
            raise e

    def _notify_tune_finished(self):
        app_id = self._get_app_id()
        self._event_queue.put(Event(event_id=app_id, event_type=Event.APP_END, event_time=datetime.datetime.now(), app_id=app_id))

    def _notify_tune_start(self):
        app_id = self._get_app_id()
        self._event_queue.put(Event(event_id=app_id, event_type=Event.APP_START, event_time=datetime.datetime.now(), app_id=app_id))

    def _notify_trial_start(self, trial):
        app_id = self._get_app_id()
        job_id = self._get_job_id(trial)
        self._event_queue.put(Event(event_id=app_id, event_type=Event.JOB_START, event_time=datetime.datetime.now(), app_id=app_id, job_id=job_id))

    def _notify_trial_end(self, trial):
        app_id = self._get_app_id()
        job_id = self._get_job_id(trial)
        self._event_queue.put(Event(event_id=app_id, event_type=Event.JOB_END, event_time=datetime.datetime.now(), app_id=app_id, job_id=job_id))


    def debug_string(self) -> str:
        """Returns a human readable message for printing to the console."""

        if self._resources_initialized:
            status = ("Resources used: {}/{} CPUs, {}/{} GPUs, "
                      "{}/{} GiB heap, {}/{} GiB objects, GPU demand: {}/{}".format(
                        self._committed_resources.cpu, self._avail_resources.cpu,
                        self._committed_resources.gpu,self._avail_resources.gpu,
                        _to_gb(self._committed_resources.memory),_to_gb(self._avail_resources.memory),
                        _to_gb(self._committed_resources.object_store_memory),_to_gb(self._avail_resources.object_store_memory),
                        self._demand.gpu, self._avail_resources.gpu,)
                      )
            return status
        else:
            return "Resources requested: ?"


    def _update_demand(self, trials: List[Trial]):
        self._demand = Resources(cpu=0,gpu=0)
        
        for trial in trials:
            if trial.status not in [Trial.TERMINATED, Trial.ERROR]:
                self._demand += trial.resources
            
        


    def on_step_begin(self, trial_runner) -> None:
        """Before step() is called, update the available resources."""



        # print(")
        msg = ["++++++++++++++++++++++++++++++"]
        msg.append(f"DEBUG: on_step_begin called. tune_started: {self._tune_started} app_id: {self._get_app_id()}")
        # msg.append(f"DEBUG: available_resources: {self._avail_resources} ray_available: {ray.available_resources()}")
        msg.append(self.debug_string())





        trials = trial_runner.get_trials()

        for trial in trials:
            msg.append(f"Trial_id: {trial.trial_id} status: {trial.status} can fulfill?: {self.has_resources_for_trial(trial)}")




        if not self._tune_started:
            self._notify_tune_start()
            self._tune_started = True



        self._update_avail_resources()

        if self._has_errored:
            self._stop_trial_runner(trial_runner)
            return

        self._update_demand(trials)

        estimated_times = trial_runner._scheduler_alg.estimate_remaining_trial_times()
        self._set_queue.put(estimated_times)

        # send a ping event quaterly w.r.t. inactivity_time
        if (datetime.datetime.now() - self._last_ping_time).total_seconds() > (self._inactivity_time/4.0):
            app_id = self._get_app_id()
            self._last_ping_time = datetime.datetime.now()
            self._event_queue.put(Event(event_id=app_id, event_type="APP_PING", event_time=self._last_ping_time, app_id=app_id))
            
        
        if self._avail_resources >= self._demand:
            if self._demand > self._committed_resources:

                msg.append("DEBUG: need to unpause cond(1)")

                """
                Unpause/unpreempt d-c trials.
                Currently not doing anything special - assuming/relying on trial_scheduler
                to keep choosing paused trials in order
                """
                pass
            elif self._demand == self._committed_resources:
                # do nothing
                pass
            else:
                raise ValueError("Demand is less than committed resources")
        else:
            if self._avail_resources > self._committed_resources:
                """
                Unpause/unpreempt r-c trials.
                Currently not doing anything special - assuming/relying on trial_scheduler
                to keep choosing paused trials in order
                """

                msg.append("DEBUG: need to unpause cond(2)")

                pass
            elif self._avail_resources == self._committed_resources:
                # do nothing
                pass
            else:
                # only point where preemption happens
                # preempt c-r trials
                running_trials = list(filter(lambda t: t.status == Trial.RUNNING, trials))

                get_trial_idx = lambda t: int(t.trial_id.split('_')[-1])
                running_trials = sorted(running_trials, key=get_trial_idx, reverse=True)
                to_preempt = self._committed_resources.gpu - self._avail_resources.gpu

                assert(to_preempt <= len(running_trials))

                # preempt the last r-c trials
                for t in range(to_preempt):
                    trial_runner._queue_decision(running_trials[t], TrialScheduler.PAUSE)
            
        self._trial_just_finished_before = self._trial_just_finished
        self._trial_just_finished = False

        self._steps += 1


        msg.append(msg[0])

        if self._steps % 10 == 0 and self._get_app_id() == 0:
            pass
            # print("\n".join(msg))


    def on_step_end(self, trial_runner) -> None:
        

        trials = trial_runner.get_trials()

        if trial_runner.is_finished():
            self._notify_tune_finished()

    def _stop_trial_runner(self, trial_runner):
        """
        This tries to stop the ongoing trial runner by setting all trial decisions to Stop
        """

        trials = trial_runner.get_trials()
        for trial in trials:
            trial_runner._queue_decision(trial, TrialScheduler.STOP)



    def has_resources(self, resources: Resources) -> bool:
        """Returns whether this runner has at least the specified resources.

        This refreshes the Ray cluster resources if the time since last update
        has exceeded self._refresh_period. This also assumes that the
        cluster is not resizing very frequently.
        """

        self._update_avail_resources()
        currently_available = Resources.subtract(self._avail_resources,
                                                 self._committed_resources)
        have_space = (
            resources.cpu_total() <= currently_available.cpu
            and resources.gpu_total() <= currently_available.gpu
            and resources.memory_total() <= currently_available.memory
            and resources.object_store_memory_total() <=
            currently_available.object_store_memory and all(
                resources.get_res_total(res) <= currently_available.get(res)
                for res in resources.custom_resources))

        if have_space:
            # The assumption right now is that we block all trials if one
            # trial is queued.
            self._trial_queued = False
            return True

        return False

    def has_resources_for_trial(self, trial: Trial) -> bool:
        """Returns whether this runner has resources available for this trial.

        Args:
            trial: Trial object which should be scheduled.

        Returns:
            boolean

        """
        return self.has_resources(trial.resources)



    def start_trial(self,
                    trial: Trial,
                    checkpoint: Optional[Checkpoint] = None,
                    train: bool = True) -> bool:



        if self.has_resources_for_trial(trial):

            prior_status = trial.status

            msg = ["++++++++++++++++++++++++++++++"]
            msg.append(f"Attempting to start trial: {trial.trial_id}")


            start_val = super(MyRayTrialExecutor, self).start_trial(trial, checkpoint, train)
            

            msg.append(f"Start trial for {trial.trial_id} Successful?: {start_val}")
            msg.append(msg[0])

            if self._steps % 10 == 0 and self._get_app_id() == 1:
                pass
                # print("\n".join(msg))


            if start_val:
                self._commit_resources(trial.resources)
                
                if prior_status == Trial.PENDING:
                    self._notify_trial_start(trial)

            return start_val
        
        return False

    def _start_trial(self, trial, checkpoint=None, runner=None,
                     train=True) -> bool:
        """Starts trial and restores last result if trial was paused.

        Args:
            trial (Trial): The trial to start.
            checkpoint (Optional[Checkpoint]): The checkpoint to restore from.
                If None, and no trial checkpoint exists, the trial is started
                from the beginning.
            runner (Trainable): The remote runner to use. This can be the
                cached actor. If None, a new runner is created.
            train (bool): Whether or not to start training.

        Returns:
            True if trial was started successfully, False otherwise.

        See `RayTrialExecutor.restore` for possible errors raised.
        """


        msg = ["++++++++++++++++++++++++++++++"]
        msg.append(f"Attempting to _start trial: {trial.trial_id}")


        successful = True

        prior_status = trial.status
        self.set_status(trial, Trial.PENDING)
        if runner is None:
            runner = self._setup_remote_runner(trial)
            if not runner:
                successful  = False
                # return False

        if successful:
            trial.set_runner(runner)
            self._notify_trainable_of_new_resources_if_needed(trial)
            self.restore(trial, checkpoint)
            self.set_status(trial, Trial.RUNNING)

            if trial in self._staged_trials:
                self._staged_trials.remove(trial)

            previous_run = self._find_item(self._paused, trial)
            if prior_status == Trial.PAUSED and previous_run:
                # If Trial was in flight when paused, self._paused stores result.
                self._paused.pop(previous_run[0])
                self._running[previous_run[0]] = trial
            elif train and not trial.is_restoring:
                self._train(trial)

        msg.append(f"Successful?: {successful} runner is None: {runner is None}")
        msg.append(msg[0])
        if self._steps % 10 == 0 and self._get_app_id() == 0:
            # print("\n".join(msg))
            pass

        return successful

    def _setup_remote_runner(self, trial):
        trial.init_logdir()
        # We checkpoint metadata here to try mitigating logdir duplication
        self.try_checkpoint_metadata(trial)
        logger_creator = partial(noop_logger_creator, logdir=trial.logdir)

        trainable_cls = trial.get_trainable_cls()
        if not trainable_cls:
            raise AbortTrialExecution(
                f"Invalid trainable: {trial.trainable_name}. If you passed "
                f"a string, make sure the trainable was registered before.")


        # print(f"DEBUG: trail {trial.trial_id} got trainable_cls: {trainable_cls}")
        _actor_cls = _class_cache.get(trainable_cls)


        full_actor_class = _actor_cls.options(
            num_cpus=trial.resources.cpu,
            num_gpus=trial.resources.gpu,
            memory=trial.resources.memory or None,
            object_store_memory=trial.resources.object_store_memory
            or None,
            resources=trial.resources.custom_resources)

        # Clear the Trial's location (to be updated later on result)
        # since we don't know where the remote runner is placed.
        trial.set_location(Location())
        logger.debug("Trial %s: Setting up new remote runner.", trial)
        # Logging for trials is handled centrally by TrialRunner, so
        # configure the remote runner to use a noop-logger.
        trial_config = copy.deepcopy(trial.config)
        trial_config[TRIAL_INFO] = TrialInfo(trial)

        stdout_file, stderr_file = trial.log_to_file
        trial_config[STDOUT_FILE] = stdout_file
        trial_config[STDERR_FILE] = stderr_file
        kwargs = {
            "config": trial_config,
            "logger_creator": logger_creator,
        }
        if issubclass(trial.get_trainable_cls(), DurableTrainable):
            kwargs["remote_checkpoint_dir"] = trial.remote_checkpoint_dir
            kwargs["sync_function_tpl"] = trial.sync_to_cloud


        # print(f"DEBUG: trial: {trial.trial_id} got actor: {full_actor_class}")

        with self._change_working_directory(trial):
            return full_actor_class.remote(**kwargs)




    # have to overload this to pass in an extra parameter to stop_trail
    def pause_trial(self, trial: Trial) -> None:
        """Pauses the trial.

        If trial is in-flight, preserves return value in separate queue
        before pausing, which is restored when Trial is resumed.
        
        We want to release resources (specifically GPUs) when pausing an
        experiment. This results in PAUSED state that similar to TERMINATED.
        """
        
        trial_future = self._find_item(self._running, trial)
        if trial_future:
            self._paused[trial_future[0]] = trial
    
        assert trial.status == Trial.RUNNING, trial.status
        try:
            self.save(trial, Checkpoint.MEMORY)
            self.stop_trial(trial, pause_only=True)
            self.set_status(trial, Trial.PAUSED)
        except Exception:
            logger.exception("Error pausing runner.")
            self.set_status(trial, Trial.ERROR)


    def stop_trial(self,
                   trial: Trial,
                   error: bool = False,
                   error_msg: Optional[str] = None,
                   pause_only: bool = False,
                   destroy_pg_if_cannot_replace: bool = True) -> None:

        super(MyRayTrialExecutor, self).stop_trial(trial, error, error_msg, destroy_pg_if_cannot_replace)
        self._return_resources(trial.resources)

        if not pause_only:
            # stop_trial was not called by pause_trial
            self._notify_trial_end(trial)


    def _stop_trial(self,
                    trial: Trial,
                    error=False,
                    error_msg=None,
                    destroy_pg_if_cannot_replace=True):
        """Stops this trial.

        Stops this trial, releasing all allocating resources. If stopping the
        trial fails, the run will be marked as terminated in error, but no
        exception will be thrown.

        If the placement group will be used right away
        (destroy_pg_if_cannot_replace=False), we do not remove its placement
        group (or a surrogate placement group).

        Args:
            error (bool): Whether to mark this trial as terminated in error.
            error_msg (str): Optional error message.

        """
        self.set_status(trial, Trial.ERROR if error else Trial.TERMINATED)
        self._trial_just_finished = True
        trial.set_location(Location())

        try:
            trial.write_error_log(error_msg)
            if hasattr(trial, "runner") and trial.runner:

                should_destroy_actor = True

                if should_destroy_actor:
                    logger.debug("Trial %s: Destroying actor.", trial)

                    with self._change_working_directory(trial):
                        self._trial_cleanup.add(trial, actor=trial.runner)

                if trial in self._staged_trials:
                    self._staged_trials.remove(trial)

        except Exception:
            logger.exception("Trial %s: Error stopping runner.", trial)
            self.set_status(trial, Trial.ERROR)
        finally:
            trial.set_runner(None)



    def _update_avail_resources(self, num_retries=5):

        if self._get_queue is not None:



            try:
                for _ in range(self._max_queue_gets):
                    resources = self._get_queue.get(block=False)

                    # convert int resources to Resource type
                    if isinstance(resources, int):

                        if resources < 0:
                            raise ValueError("Force stop")

                        # dont know what will happen if we don't do this
                        resources = Resources(cpu=resources, gpu=resources)
                        

                    if not isinstance(resources, Resources):
                        raise ValueError(f"resources not of type Resources")
                    # print(f"Got resources: {resources}")

                    self._avail_resources = resources

            except Empty:
                # do nothing. no need to update
                pass
            except Exception as e:
                print(f"Error: {e}")
                self._has_errored = True


            self._last_resource_refresh = time.time()
            self._resources_initialized = True

        else:
            raise ValueError("self._get_queue is not initialized")