import ray
from ray.util.queue import Queue
import os, sys

from common import Event, App, Job
from GenericScheduler import AppGenericScheduler


from datetime import datetime, timedelta
from time import sleep
import random

class RayAppGenericScheduler(AppGenericScheduler):
    """docstring for RayAppGenericScheduler"""
    def __init__(self, total_gpus, event_queue, app_list, app_info_fn="results.csv", suppress_print=True, estimate=True):


        super(RayAppGenericScheduler, self).__init__(int(total_gpus), event_queue, app_list, app_info_fn, suppress_print)
    
        self._inactivity_time=1440
        self._extra_service = 200
        self._maximum_events_to_process = 10
        self._estimate = estimate

    def sim_estimate(self, app):
        raise NotImplementedError


    def start_app(self, app, event_time):
        
        if app.status == App.SUBMITTED or app.status == App.QUEUED:

            app.trial_runner_queue = {"downlink": Queue(), "uplink": Queue()}



            app.future = app.exec_func.remote(app, self._event_queue,
                                inactivity_time=self._inactivity_time)

            print(f"app_id={app.app_id} successfully started: {app.future}")

    def update_allocations(self, event_time):
        super(RayAppGenericScheduler, self).update_allocations(event_time)

        for app in self._active_apps:
            
            if hasattr(app, "trial_runner_queue"):
                if app.status != App.FAILED:
                    app.trial_runner_queue["downlink"].put(int(app.allocation))

    def update_remaining_service(self, event_time, app):

                
        if app.status == App.ACTIVE:


            num_gets = len(app.trial_runner_queue["uplink"])
            

            trial_id_to_estimated_remaining_time = {}
            for _ in range(num_gets):
                trial_id_to_estimated_remaining_time = app.trial_runner_queue["uplink"].get(block=False)

            for trial_id in trial_id_to_estimated_remaining_time:
                
                job_id = int(trial_id.split("_")[-1])
                app.jobs[job_id].remaining_service = trial_id_to_estimated_remaining_time[trial_id]


            app_ray_remaining_service = 0
            for job in app.jobs.values():
                app_ray_remaining_service += job.remaining_service

            app.remaining_service = app_ray_remaining_service

            '''
            print(f"app.remaining_service: {app.remaining_service} app_ray_remaining_service: {app_ray_remaining_service}")
            timedelta = ((event_time - self._last_event_time).total_seconds())
            app.remaining_service -= app.allocation * timedelta
            '''


            '''
            for job in app.jobs.values():
                if job.status == Job.ACTIVE:

                    prev_remaining_service = job.remaining_service

                    job.remaining_service -= job.allocation * timedelta
                    

                    # print(f"DEBUG: app_id: {job.app_id} job_id: {job.job_id} prev_remaining_service: {prev_remaining_service} remaining_service: {job.remaining_service} allocation: {job.allocation} timedelta: {timedelta}")
            # print(f"================{(event_time - self._init_time).total_seconds()}==============================")
            '''
    

    def handle_app_sub_event(self, event):
        super(RayAppGenericScheduler, self).handle_app_sub_event(event)
        app = self._app_list[event.app_id]
        app.last_event_time = event.event_time
    


    def handle_app_start_event(self, event):
        app = self._app_list[event.app_id]
        app.start_time = event.event_time
        app.on_app_start(event.event_time)
        app.last_event_time = event.event_time


    def handle_job_start_event(self, event):
        app = self._app_list[event.app_id]
        job = app.jobs[event.job_id]
        job.start_time = event.event_time
        app.last_event_time = event.event_time

        # app.update_allocation(min(int(self._app_id_to_allocation[app.app_id]), app.demand))

    def handle_job_end_event(self, event):
        app = self._app_list[event.app_id]        
        job = app.jobs[event.job_id]
        app.on_job_end(job.job_id, event.event_time)
        self._num_finished_jobs += 1
        app.last_event_time = event.event_time

        # job_statuses = [j.status == Job.END for j in app.jobs.values()]
        # if all(job_statuses):
        #     app.status = App.PENDING_END


    def handle_app_ping_event(self, event):
        app = self._app_list[event.app_id]
        app.last_event_time = event.event_time



    def handle_app_end_event(self, event):

        app = self._app_list[event.app_id]
        
        app.status = App.END


        app.end_time = event.event_time
        

        # remove from active_apps
        for i, a in enumerate(self._active_apps):
            if a.app_id == app.app_id:
                self._active_apps.pop(i)
                break
            
        self._num_finished_apps += 1

        if not self._suppress_print:
            print("active_apps: %d \t Apps done: %d" % \
                (len(self._active_apps), self._num_finished_apps))

        self.log_app_info(app)
        app.last_event_time = event.event_time

        for attr in ['future', 'exec_func', 'trial_runner_queue']:
            if hasattr(app, attr):
                delattr(app, attr)



    def restart_failed_app(self, app):
        
        for job in app.jobs.values():
            job.__init__(app_id=job.app_id, job_id=job.job_id,  service=job.service, demand=1)
        app.__init__(app_id=app.app_id, jobs=app.jobs)

        self._event_queue.put(Event(event_id=app.app_id, event_time=datetime.now(), event_type=Event.APP_SUB, app_id=app.app_id))



    def remove_failed_app(self, app):

        app.status = App.FAILED
        
        app.end_time = datetime.now()
        
        # remove from active_apps
        for i, a in enumerate(self._active_apps):
            if a.app_id == app.app_id:
                self._active_apps.pop(i)
                break
            
        self._num_finished_apps += 1
        app.last_event_time = datetime.now()

        ray.cancel(app.future, force=True, recursive=True)

        for attr in ['future', 'exec_func', 'trial_runner_queue']:
            if hasattr(app, attr):
                delattr(app, attr)


    def remove_failed_apps(self):
        
        print(f"checking for failed apps")

        for app in self._active_apps:
            if app.status == App.ACTIVE:
               
                if (datetime.now() - app.last_event_time).total_seconds() > self._inactivity_time:

                    print(f'***********KILLING APP_{app.app_id} time: {datetime.now()}************')

                    app.trial_runner_queue["downlink"].put(-1)
                    self.remove_failed_app(app)


                elif app.remaining_service < -1.0 * self._extra_service:
                    pass



    def report_progress(self):

        print("++++++++++++++++++++++++++++++++++++++++")

        '''
        if hasattr(event, "app_id"):
            print(f"event.app_id: {event.app_id}")
        print(f"event.event_type: {event.event_type}")
        print(f"event.event_time: {(event.event_time - self._init_time).total_seconds()}")
        '''

        print(f"time: {datetime.now()} elapsed_time: {self.absolute_time(datetime.now())}")
        print(f"self._num_finished_apps: {self._num_finished_apps} len(self._active_apps): {len(self._active_apps)}") 
        
        for app in self._active_apps:
            print(f"app.app_id: {app.app_id} app.status: {app.status} app.demand: {app.demand} app.allocation: {app.allocation} app.remaining_service: {int(app.remaining_service)}")

            '''
            for job in app.jobs.values():
                print(f"job_id: {job.job_id} job.remaining_service: {job.remaining_service}")
            '''
        print("++++++++++++++++++++++++++++++++++++++++")


    def process_event(self, event):

        self.progress_active_apps(event.event_time)
        self._last_event_time = event.event_time

        # if self._app_list[event.app_id].status == App.FAILED:
        #     return

        if event.event_type == Event.APP_SUB:


            
            if self._estimate and random.uniform(0,1) < 1.2:
                self.sim_estimate(app = self._app_list[event.app_id])

            self.handle_app_sub_event(event)
        elif event.event_type == Event.APP_START:
            self.handle_app_start_event(event)
        elif event.event_type == Event.APP_END:
            self.handle_app_end_event(event)
        elif event.event_type == Event.JOB_START:
            self.handle_job_start_event(event)
        elif event.event_type == Event.JOB_END:
            self.handle_job_end_event(event)
        elif event.event_type == "APP_PING":
            self.handle_app_ping_event(event)

    @ray.remote
    def gen_background_event(event_queue, event, sleep_time):
        sleep(sleep_time)
        event.event_time = datetime.now()
        event_queue.put(event)

    def run(self):


        if self._app_info_fn:
            with open(self._app_info_fn,'w') as fp:
                fp.write("app_id,submit_time,start_time,end_time,estimated_start_time,estimated_end_time,fair_act,service,num_apps_seen_diff\n")


        last_self_check_time = datetime.now()

        while self._num_finished_apps < len(self._app_list):
            
            # events = list()
            # num_gets = len(self._event_queue)

            resource_change_event = False

            if not self._event_queue.empty():

                num_gets = min(self._maximum_events_to_process, len(self._event_queue))

                for _ in range(num_gets):
                    event = self._event_queue.get(block=True)
                    self.process_event(event)
                    
                    if event.event_type in [Event.APP_SUB, Event.JOB_END]:
                        resource_change_event = True


                    print(f"DEBUG: process_event: {event.event_type} len(self._event_queue): {len(self._event_queue)} num_gets: {num_gets}")
            else:

                print(f"waiting on event")
                event = self._event_queue.get(block=True)
                self.process_event(event)

                if event.event_type in [Event.APP_SUB, Event.JOB_END]:
                    resource_change_event = True

        

            if (datetime.now() - last_self_check_time).total_seconds() > self._inactivity_time:
                self.remove_failed_apps()
                last_self_check_time = datetime.now()
                resource_change_event = True
                
                # if len(self._event_queue) <= 5:

            if resource_change_event:
                self.update_allocations(datetime.now())

            self.report_progress()
