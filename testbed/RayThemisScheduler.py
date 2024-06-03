import ray
from RayGenericScheduler import RayAppGenericScheduler
import os, sys
from datetime import datetime, timedelta
from time import sleep
import math
from functools import partial
import copy

from ThemisScheduler import AppThemisScheduler
from common import Event, App, Job

class RayAppThemisScheduler(RayAppGenericScheduler):
    """docstring for RayAppMCScheduler"""
    def __init__(self, total_gpus, event_queue, app_list, quantum=120, app_info_fn="results.csv", suppress_print=False, estimate=True):
        super(RayAppThemisScheduler, self).__init__(total_gpus, event_queue, app_list, app_info_fn, suppress_print=suppress_print, estimate=estimate)
        self._quantum = quantum


    def sim_estimate(self, app):


        # total_gpus, event_queue, app_list, class_detail, app_info_fn="results.csv", suppress_print=False, estimate=False

        snap_shot = AppThemisScheduler(total_gpus=self._max_capacity,
                                    event_queue=[Event(event_id=app.app_id, event_time=datetime.now(), event_type=Event.APP_SUB, app_id=app.app_id)],
                                    app_list={},
                                    quantum=self._quantum,
                                    app_info_fn=None)


        snap_shot._active_apps = copy.deepcopy(self._active_apps)
        snap_shot._app_id_to_allocation = copy.deepcopy(self._app_id_to_allocation)

        for virtual_app in snap_shot._active_apps+[copy.deepcopy(app)]:
            snap_shot._app_list[virtual_app.app_id] = virtual_app

        snap_shot._suppress_print = True
        snap_shot._verbosity = 0
        snap_shot._init_time = self._init_time
        snap_shot._last_event_time = self._last_event_time

        snap_shot.update_end_events(datetime.now())

        
        def break_cond(v_app):
            if v_app.status == App.END:
                return True
            return False
        
        # snap_shot.run()

        snap_shot.run(partial(break_cond, snap_shot._app_list[app.app_id]))
        

        app.update_estimates(snap_shot._app_list[app.app_id].start_time,
                            snap_shot._app_list[app.app_id].end_time)


    def valuation_function(self, app, event_time, max_allocation):

        observed_contention = app.num_apps_seen[0]/app.num_apps_seen[1]

        if max_allocation > 0:
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


    def run(self):

        @ray.remote
        def gen_background_event(event_queue, event, sleep_time):
            sleep(sleep_time)
            event.event_time = datetime.now()
            event_queue.put(event)
            
        if self._app_info_fn:
            with open(self._app_info_fn,'w') as fp:
                fp.write("app_id,submit_time,start_time,end_time,estimated_start_time,estimated_end_time,fair_act,service,num_apps_seen_diff\n")



        last_self_check_time = datetime.now()

        while self._num_finished_apps < len(self._app_list):
        
            resource_change_event = False
            redivision_event = False

            if not self._event_queue.empty():

                num_gets = min(self._maximum_events_to_process, len(self._event_queue))

                for _ in range(num_gets):
                    event = self._event_queue.get(block=True)
                    self.process_event(event)
                    
                    if event.event_type in [Event.APP_SUB, Event.JOB_END]:
                        resource_change_event = True
                    if event.event_type == "REDIVISION":
                        redivision_event = True

                     
            else:
                event = self._event_queue.get(block=True)
                self.process_event(event)

                if event.event_type in [Event.APP_SUB, Event.JOB_END]:
                    resource_change_event = True
                if event.event_type == "REDIVISION":
                    redivision_event = True

        

            if (datetime.now() - last_self_check_time).total_seconds() > self._inactivity_time:
                self.remove_failed_apps()
                last_self_check_time = datetime.now()
                resource_change_event = True

            
            if resource_change_event or redivision_event:
                self.update_allocations(event.event_time)
            if redivision_event:

                next_redivision = self._quantum
                assert(float(next_redivision) >= 0)

                gen_background_event.remote(self._event_queue,
                                        Event(event_id=None, event_type="REDIVISION", event_time=None),
                                        float(next_redivision))

            self.report_progress()