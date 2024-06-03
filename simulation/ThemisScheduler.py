import sys
import os
import copy
from datetime import datetime, timedelta
import math
import random
import ray
from GenericScheduler import AppGenericScheduler
from GenericScheduler import multi_runner
from common import Event, App, Job



class AppThemisScheduler(AppGenericScheduler):
    """docstring for AppThemisScheduler"""
    def __init__(self, total_gpus, event_queue, app_list, quantum=1, app_info_fn="results.csv", suppress_print=False, verbosity=1, p_error=None):
        super(AppThemisScheduler, self).__init__(total_gpus, event_queue, app_list, app_info_fn, suppress_print, verbosity, p_error)

        self._redivision_event = None
        self._quantum = quantum

                
    def valuation_function(self, app, event_time, max_allocation):


        if max_allocation > 0:
            observed_contention = app.num_apps_seen[0]/app.num_apps_seen[1]
            divided_cluster_size = self._max_capacity/observed_contention
            t_id = app.service/min(divided_cluster_size, app.initial_demand)
            t_sh = (event_time - app.submit_time).total_seconds() + (app.remaining_service/max_allocation)
            return t_sh/t_id
        else:
            return 1.0

    def compute_allocation(self, event_time):
        
        # self._active_apps = sorted(self._active_apps, key=lambda a: self.valuation_function(a, event_time), reverse=True)

        total_demand = sum([app.demand for app in self._active_apps])

        residual = min(total_demand, self._max_capacity)

        app_id_to_allocation = {}

        app_slice = self._active_apps[:]


        for app in self._active_apps:
            app_id_to_allocation[app.app_id] = 0


        while residual > 0:

            app_slice = sorted(app_slice, key=lambda a: self.valuation_function(a, event_time, min(residual, a.demand)), reverse=True)

            top_app = app_slice[0]
            allocation = min(residual, top_app.demand)

            app_id_to_allocation[top_app.app_id] = allocation
            residual -= allocation

            app_slice = app_slice[1:]

        assert(math.isclose(residual, 0)), residual

        return app_id_to_allocation

    def __pick_min_heap(self, heap1, heap2):

        if len(heap1) == 0:
            return heap2
        elif len(heap2) == 0:
            return heap1
        
        heap1_event = heap1[0]
        heap2_event = heap2[0]

        if heap1_event < heap2_event:
            return heap1
        else:
            return heap2


    def redivision(self, event):
        self._redivision_event = None

        next_redivision = self._quantum
        
        self._redivision_event = Event(event_id=0, event_time=event.event_time + timedelta(seconds=float(next_redivision)), event_type="REDIVISION")



    def __pick_min_event(self):

        numbers = [self._closest_end_event, self._redivision_event]
        lst = self._event_queue

        inf_event = Event(event_id=-1, event_time=datetime.max, event_type=Event.UNDEFINED)


        numbers = [n if n else inf_event for n in numbers]
        min_answer = min(min(numbers), lst[-1] if lst else inf_event)

        if lst and min_answer.event_type == Event.APP_SUB:
            lst.pop()
        return min_answer


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

            if event.event_type in [Event.APP_SUB, Event.JOB_END, "REDIVISION"]:
                self.redivision(event)

            self.update_allocations(event.event_time)

            self.update_end_events(event.event_time)

        
            if event.event_type == Event.APP_SUB and self._estimator and random.uniform(0,1) < p_of_estimate:

                if self._collect_dataset_stats_flag:
                    self.collect_dataset_stats(self._app_list[event.app_id])
                
                ret = self.sim_estimate(app=self._app_list[event.app_id], event_time=event.event_time)
                if ret:
                    self._sim_futures.append(ret)
                
            # print(f"AFTER: event_type: {event.event_type} len(event_queue): {len(self._event_queue)} self._closest_end_event: {self._closest_end_event.event_time if self._closest_end_event else None} len(active_apps): {len(self._active_apps)}")

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
