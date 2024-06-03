import random
import os, sys
from datetime import datetime, timedelta
import copy
import pickle
from functools import partial
import ray
import math
from time import sleep

from common import Event, App, Job


@ray.remote
def multi_runner(batch_snap_shots):
    futures = list()
    for batch in batch_snap_shots:
        estimator, app_id, event_time = batch
        futures.append(scheduler_run_ray.remote(estimator, app_id, event_time))
    return futures



@ray.remote
def scheduler_run_ray(snap_shot, app_id, event_time):
    


    snap_shot.update_end_events(event_time)
    
    def break_cond(v_app):
        if v_app.status == App.END:
            return True
        return False


    snap_shot.run(partial(break_cond, snap_shot._app_list[app_id]))
    
    return app_id, snap_shot._app_list[app_id].start_time, snap_shot._app_list[app_id].end_time

class AppGenericScheduler(object):
    """This class implements a Generic Scheduler for apps"""

    def __init__(self, total_gpus, event_queue, app_list, app_info_fn="results.csv", suppress_print=False, verbosity=1, p_error=None):
        
        self._max_capacity = total_gpus
        self._avail_capacity = total_gpus
        
        self._active_apps = list()
        
        self._num_finished_jobs = 0
        self._num_finished_apps = 0


        self._app_id_to_allocation = None

        self._event_queue = event_queue
        self._closest_end_event = None
        self._app_list = app_list
        
        self._estimator = None
        self._estimate_batch = 100
        self._suppress_print = suppress_print
        self._verbosity = verbosity
        

        self._gpu_util = {}
        self._stats_timeline_ticks = list()
        self._stats_timeline_gpu_util = list()
        self._stats_timeline_queue_length = list()


        self._init_time = datetime.now()
        self._last_event_time = datetime.now()
        self._app_info_fn = app_info_fn


        # extra features - not core to scheduling
        self._p_error = p_error


        # collecting dataset for trianing estimator
        self._training_dataset_pkl = None
        self._collect_dataset_stats_flag = False

        self._snap_shots = list()
        self._sim_futures = None
        self._ray_queue = None

        self._perf_timer = []


        '''
        class EWMA(object):
            """docstring for EWMA"""
            def __init__(self, avg):
                super(EWMA, self).__init__()
                self.avg = avg
                self.weight_new_sample = 0.25
            def update(self, new_sample):
                self.avg = self.avg*(1.0 - self.weight_new_sample) + (new_sample * self.weight_new_sample)

        self._avg_contention = EWMA(0)
        '''



    def set_estimator(self):
        self._estimator = copy.deepcopy(self)
        self._estimator._active_apps = None
        self._estimator._app_list = None
        self._estimator._event_queue = list()
        self._estimator._estimator = None
        self._estimator._suppress_print = True
        self._estimator._verbosity = 0
        self._estimator._app_info_fn = None


    def __snap_shot(self):
        self._estimator._active_apps = self._active_apps
        self._estimator._last_event_time = self._last_event_time
        self._estimator._app_list = {}
        
        for app in self._estimator._active_apps:
            self._estimator._app_list[app.app_id] = app

        return copy.deepcopy(self._estimator)


    def collect_dataset_stats(self, app):

        # assert(app.status == App.SUBMITTED), app.status


        app.training_example = {"id": app.app_id}
        app.training_example["service"] = app.remaining_service
        


        ll = []
        for a in self._active_apps:
            if a.app_id == app.app_id:
                continue
            ll.append((a.remaining_service, a.app_id))

        app.training_example[f"current"] = ll


    def absolute_time(self, dtime):
        return (dtime - self._init_time).total_seconds()


    def compute_allocation(self):
        raise NotImplementedError

    @property
    def available_capacity(self):
        return self._available_capacity
            

    # always an error on the true job remaining service times
    def estimate_app_service_time(self, app):
        if self._p_error and self._estimator:

            if app.induced_error == 0:
                app.induced_error = (1.0 + random.uniform(-1.0*(self._p_error/100.0),(self._p_error/100.0)))
            
            for job in app.jobs.values():
                job.estimated_remaining_service = max(job.remaining_service*app.induced_error, 1e-2)
        else:
            for job in app.jobs.values():
                job.estimated_remaining_service = job.remaining_service

        # make decisions based on estimates. counts based on actual ones            
        # job.estimated remaining service times have been updated as well
        app.estimated_remaining_service = sum([job.estimated_remaining_service for job in app.jobs.values()])
        app.estimated_service = app.estimated_remaining_service

    def update_remaining_service(self, event_time, app):
        
        if app.status == App.ACTIVE:

            for job in app.jobs.values():
                
                # print(f"app_id: {app.app_id} job.allocation: {job.allocation} job.thrpt(job.allocation): {job.thrpt(job.allocation)}")

                job.remaining_service -= job.thrpt(job.allocation) * (event_time - self._last_event_time).total_seconds()
                app.remaining_service -= job.thrpt(job.allocation) * (event_time - self._last_event_time).total_seconds()
                
                self.estimate_app_service_time(app)

                assert(job.remaining_service >= -1e6 ), job.remaining_service
                    
    def progress_active_apps(self, event_time):    

        for app in self._active_apps:
            self.update_remaining_service(event_time, app)
            app.num_apps_seen = (app.num_apps_seen[0]+len(self._active_apps), app.num_apps_seen[1]+1)

    def update_allocations(self, event_time):
        

        # self._avg_contention.update(len(self._active_apps))
        
        self._app_id_to_allocation = self.compute_allocation(event_time)

        self._avail_capacity = self._max_capacity

        for app in self._active_apps:

            # (re)start_app, simply change rate, preempt_app

            # app.update_allocation(min(int(self._app_id_to_allocation[app.app_id]), app.demand))
            app.update_allocation(min((self._app_id_to_allocation[app.app_id]), app.demand))


            if app.status == App.SUBMITTED:
                
                if app.allocation > 0:
                    self.start_app(app, event_time)
                    app.status = App.ACTIVE
                else:
                    app.status = App.QUEUED

            elif app.status == App.ACTIVE:
                
                if app.allocation == 0:
                    app.status = App.PREEMPTED

            elif app.status == App.QUEUED:
                
                if app.allocation > 0:
                    self.start_app(app, event_time)
                    app.status = App.ACTIVE

            elif app.status == App.PREEMPTED:

                if app.allocation > 0:
                    self.start_app(app, event_time)
                    app.status = App.ACTIVE
            else:
                pass
            
            self._avail_capacity -= app.allocation

            
    def update_end_events(self, event_time):

        # assuming app.allocation is the most recent one and individual jobs have been assigned a rate

        

        
        self._closest_end_event = None

        for app in self._active_apps:

            for job in app.jobs.values():

                if job.status == Job.END:
                    continue

                projected_end_time = datetime.max

                if math.isclose(job.allocation, 0) and not math.isclose(job.remaining_service, 0):
                    projected_end_time = datetime.max
                else:

                    # if math.isclose(job.allocation, 0) and math.isclose(job.remaining_service, 0):
                        
                    #     job.remaining_service = 0
                    #     job.attempts[-1]["end_time"] = event_time

                    if math.isclose(job.remaining_service, 0):

                        job.remaining_service = 0
                        projected_end_time = event_time

                    else:    

                        try:
                            projected_end_time = event_time + timedelta(seconds = job.remaining_service/job.thrpt(job.allocation))
                        except Exception as e:
                            projected_end_time = datetime.max
                        

                    # if len(job.attempts) > 0:
                    #     job.attempts[-1]["end_time"] = projected_end_time
                    # else:
                    #     job.attempts.append({"end_time": projected_end_time})

                    job.end = projected_end_time

                    event = Event(event_id=job.job_id, event_time=job.end,
                                event_type=Event.JOB_END, app_id=app.app_id, job_id=job.job_id)

                    if not self._closest_end_event:
                        self._closest_end_event = event
                    else:
                        self._closest_end_event = event if event < self._closest_end_event else self._closest_end_event

    def handle_app_sub_event(self, event):
        
        
        app = self._app_list[event.app_id]

        app.status = App.SUBMITTED
        app.on_app_submit(event.event_time)
        app.submit_time = event.event_time

        self._active_apps.append(app)

        app.num_apps_seen = (len(self._active_apps), 1)

        self.estimate_app_service_time(app)




    # have to look at this
    def start_app(self, app, event_time):
        
        if app.status == App.SUBMITTED or app.status == App.QUEUED:
            app.start_time = event_time

            '''            
            for job in app.jobs.values():

                if job.allocation > 0:
                    projected_end_time = event_time + timedelta(seconds=job.remaining_service/job.thrpt(job.allocation))
                else:
                    projected_end_time = datetime.max

                event = Event(event_id=job.job_id, event_time=projected_end_time,
                            event_type=Event.JOB_END, app_id=app.app_id, job_id=job.job_id)
            '''
                
                
        
        app.on_app_start(event_time)



    def log_app_info(self, app):

        if self._app_info_fn == None or app.status == App.FAILED:
            return



        submit_time = (app.submit_time - self._init_time).total_seconds()
        start_time = (app.start_time - self._init_time).total_seconds()
        end_time = (app.end_time - self._init_time).total_seconds()


        num_apps_seen_diff = app.num_apps_seen[0]/app.num_apps_seen[1]


        divided_cluster_size = self._max_capacity/num_apps_seen_diff
        fair_act = app.service/min(divided_cluster_size, app.initial_demand)

        if len(app.estimated_start_time) == 0:
            estimated_start_time = -1
            estimated_end_time = -1
        else:            
            estimated_start_time = (app.estimated_start_time[0] - self._init_time).total_seconds()
            estimated_end_time = (app.estimated_end_time[0] - self._init_time).total_seconds()


        if self._training_dataset_pkl != None:
            app.training_example["ACT"] = end_time - submit_time


            ll=[]
            for app_id in self._app_list:
                a = self._app_list[app_id]
                # if a.submit_time >= app.submit_time and a.submit_time <app.end_time and a.status != App.UNDEFINED:
                if a.submit_time >= app.submit_time and a.submit_time <app.end_time and a.status != App.UNDEFINED and a.app_id != app.app_id:
                    ll.append(((a.submit_time - app.submit_time).total_seconds(), a.service, a.app_id))
            
            ll = sorted(ll, key=lambda e: e[0])
            app.training_example["future"] = ll 


            with open(self._training_dataset_pkl, 'ab') as fp:
                pickle.dump(app.training_example, fp)


        with open(self._app_info_fn, 'a') as fp:
            fp.write(f"{app.app_id},{submit_time},{start_time},{end_time},{estimated_start_time},{estimated_end_time},{fair_act},{app.service},{num_apps_seen_diff}\n")


    def sim_estimate(self, app, event_time):


        # tick = datetime.now()


        # promise = self._ray_queue.put_async([self._estimator,app.app_id,event_time])
        # promise = self._ray_queue.put([self._estimator,app.app_id,event_time])
        
        # self._snap_shots.append(ray.put([self._active_apps,app.app_id,event_time]))
        # self._snap_shots.append([self._active_apps,app.app_id,event_time])
        
        estimator = self.__snap_shot()

        self._snap_shots.append([estimator,app.app_id,event_time])
        
        # pickle.dump([self._active_apps,app.app_id,event_time,self._last_event_time], self._pp)
        # self._ray_queue.put()

        # f = scheduler_run_ray.remote(self._estimator,app.app_id,event_time)
        # tock = datetime.now()
                

        # self._perf_timer.append([len(self._active_apps),(tock-tick).total_seconds()])
        
        
        if len(self._snap_shots) == self._estimate_batch:
            future = multi_runner.remote(self._snap_shots)
            self._snap_shots = list()
            return future
        return None
        
    def sim_estimate_old(self, app):

        temp_event_queue = self._event_queue
        temp_app_list = self._app_list

        self._event_queue = list()
        self._app_list = {}

        snap_shot = copy.deepcopy(self)

        for virtual_app in snap_shot._active_apps:
            snap_shot._app_list[virtual_app.app_id] = virtual_app

        snap_shot._estimate = False
        snap_shot._suppress_print = True
        snap_shot._verbosity = 0

        def break_cond(v_app):
            if v_app.status == App.END:
                return True
            return False


        snap_shot.run(partial(break_cond, snap_shot._app_list[app.app_id]))
        
        app.update_estimates(snap_shot._app_list[app.app_id].start_time,
                            snap_shot._app_list[app.app_id].end_time)

        self._event_queue = temp_event_queue
        self._app_list = temp_app_list


    def handle_job_end_event(self, event):
        

        app = self._app_list[event.app_id]
        job = app.jobs[event.job_id]


        # assert(math.isclose(job.remaining_service, 0.0, abs_tol=0.01)), (self._active_apps[-1].app_id)
        assert(math.isclose(job.remaining_service, 0.0, abs_tol=0.01)), (job.remaining_service)
        

        app.on_job_end(job.job_id, event.event_time)
        

        self._num_finished_jobs += 1

        

        job_statuses = [j.status == Job.END for j in app.jobs.values()]

        # all_jobs_have_finished
        if all(job_statuses):

            # stats for gpu util
            # self.log_clusterlog_app_info_stats(event.event_time)


            app.status = App.END
            app.end_time = event.event_time

            # remove from active_apps
            for i, a in enumerate(self._active_apps):
                if a.app_id == app.app_id:
                    # self._active_apps.pop(i)
                    del self._active_apps[i]
                    break
                
            self._num_finished_apps += 1



    def __pick_min_event(self):


        if len(self._event_queue) == 0:
            return self._closest_end_event
        elif not self._closest_end_event:
            return self._event_queue.pop()

        new_event = self._event_queue.pop()

        if new_event < self._closest_end_event:
            return new_event

        self._event_queue.append(new_event)
        return self._closest_end_event



    def run(self, cond=lambda: False):



        if self._app_info_fn != None and not self._suppress_print:

            if self._collect_dataset_stats_flag:
                self._training_dataset_pkl = self._app_info_fn.replace('.csv', '.pkl')

                with open(self._training_dataset_pkl, 'wb') as fp:
                    pass

            with open(self._app_info_fn,'w') as fp:
                fp.write("app_id,submit_time,start_time,end_time,estimated_start_time,estimated_end_time,fair_act,service,num_apps_seen_diff\n")




        if self._estimator:



            p_of_estimate = min(5000.0/len(self._app_list), 1.0)


            if not ray.is_initialized():
                if ray.__version__ == '2.0.0.dev0':
                    ray.init(ignore_reinit_error=True, address="auto")
                elif ray.__version__ == '2.10.0':
                    ray.init(ignore_reinit_error=True, address="auto", runtime_env={"env_vars": {"PYTHONPATH": "${PYTHONPATH}:"+f"{os.path.dirname(__file__)}/"}})
                else:
                    print("Warning: Untested Ray version --- may result in erroneous behaviour")

            self._sim_futures = list()

        while len(self._event_queue) > 0 or self._closest_end_event:

            event = self.__pick_min_event()

            self.progress_active_apps(event.event_time)            
            self._last_event_time = event.event_time


            self.report_progress(event)


            if event.event_type == Event.APP_SUB:
                self.handle_app_sub_event(event)
            elif event.event_type == Event.JOB_END:
                self.handle_job_end_event(event)


            self.update_allocations(event.event_time)

            self.update_end_events(event.event_time)

            # ray changes here
            if event.event_type == Event.APP_SUB and self._estimator and random.uniform(0,1) < p_of_estimate:

                if self._collect_dataset_stats_flag:
                    self.collect_dataset_stats(self._app_list[event.app_id])
            
                
                ret = self.sim_estimate(app=self._app_list[event.app_id], event_time=event.event_time)
                if ret:
                    self._sim_futures.append(ret)
                

            if cond():
                break

        # ray changes here
        if self._estimator:


            
            if len(self._snap_shots) > 0:
                self._sim_futures.append(multi_runner.remote(self._snap_shots))

            batched_futures = ray.get(self._sim_futures)
            futures = []
            for b in batched_futures:
                futures += b

            total_tasks = len(futures)
            finished_futures = list()
            
            while futures:
                finished, futures = ray.wait(futures)
                finished_futures += finished
            
            for future in finished_futures:            
                app_id, estimated_start_time, estimated_end_time = ray.get(future)
                self._app_list[app_id].update_estimates(estimated_start_time, estimated_end_time)
                if self._verbosity == 4:
                    print(f"num ray finished: {total_tasks-len(futures)}", end='\r')
            
        self.log_apps()

    def util_print_progress(self, event):
        print(f"event_type: {event.event_type} event_time: {(event.event_time - self._init_time).total_seconds()}")
        for app in self._active_apps:
            print(f"app_id: {app.app_id} allocation: {app.allocation} app.demand: {app.demand} app.remaining_service: {app.remaining_service}")
            for job in app.jobs.values():
                print(f"\tjob_id: {job.job_id} allocation: {job.allocation} status: {job.status} job.remaining_service: {job.remaining_service}")
        print("==============================")


    def log_apps(self):

        if self._app_info_fn:
            for app_id in self._app_list:
                self.log_app_info(self._app_list[app_id])
               


    def util_pickle_progress(self, event):

        log_fname = self._app_info_fn.replace(".csv", ".pkl")

        with open(log_fname, 'ab') as fp:

            pickle.dump([event, self._active_apps, self._init_time], fp)

     
    def report_progress(self, event):
        if self._verbosity == 1:
            pass
            print("\ractive_apps: %d \t Apps done: %d" % (len(self._active_apps), self._num_finished_apps),end='')
        elif self._verbosity == 2:
            self.util_print_progress(event)
        elif self._verbosity == 3:
            self.util_pickle_progress(event)
        elif self._verbosity == 4:
            self.util_print_progress(event)
            self.util_pickle_progress(event)                


    @property
    def total_gpus(self):
        return self._avail_capacity

    @property
    def max_gpus(self):
        return self._max_capacity
    

    @property
    def num_finished_jobs(self):
        return self._num_finished_jobs    

    @property
    def num_finished_apps(self):
        return self._num_finished_apps