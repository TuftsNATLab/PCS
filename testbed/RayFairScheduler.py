import ray
from RayGenericScheduler import RayAppGenericScheduler
import os, sys
from datetime import datetime, timedelta
from time import sleep
import numpy as np
from functools import partial
import copy
from fractions import Fraction as frac


from common import Event, App, Job
from FairScheduler import AppFairScheduler


class RayAppFairScheduler(RayAppGenericScheduler):
    """docstring for RayAppFairScheduler"""
    def __init__(self, total_gpus, event_queue, app_list, quantum=100, app_info_fn="results.csv", suppress_print=False, estimate=True):

        super(RayAppFairScheduler, self).__init__(total_gpus, event_queue, app_list, app_info_fn, suppress_print=suppress_print, estimate=estimate)

        self._quantum = quantum
        self._min_quantum = quantum
        self._app_id_to_fair_allocation = {}
        self._app_id_to_fractional_allocation = {}
        self._app_id_to_int_allocation = {}
        self._fractional_share = None

    def sim_estimate(self, app):


        snap_shot = AppFairScheduler(total_gpus=self._max_capacity,
                                    event_queue=[Event(event_id=app.app_id, event_time=datetime.now(), event_type=Event.APP_SUB, app_id=app.app_id)],
                                    app_list={},
                                    app_info_fn=None)


        snap_shot._active_apps = copy.deepcopy(self._active_apps)
        snap_shot._app_id_to_allocation = copy.deepcopy(self._app_id_to_allocation)

        

        # create app_list
        for virtual_app in snap_shot._active_apps+[copy.deepcopy(app)]:
            snap_shot._app_list[virtual_app.app_id] = virtual_app


        snap_shot._suppress_print = True
        snap_shot._verbosity = 0
        snap_shot._init_time = self._init_time
        snap_shot._last_event_time = self._last_event_time

        # snap_shot.update_end_events(datetime.now())

        
        def break_cond(v_app):
            if v_app.status == App.END:
                return True
            return False
        
        # snap_shot.run()

        snap_shot.run(partial(break_cond, snap_shot._app_list[app.app_id]))
        

        app.update_estimates(snap_shot._app_list[app.app_id].start_time,
                            snap_shot._app_list[app.app_id].end_time)


    def compute_allocation(self, event_time):

        # compute fair and guarantee int(fair-share) to every app
        # app_id_to_fair_allocation = self.compute_fair_allocation()

        '''
        print("=====================================")
        print(f"residual: {residual}")
        for app in self._active_apps:
            print(f"app_id: {app.app_id} int_allocation: {self._app_id_to_int_allocation[app.app_id]} fair_allocation: {self._app_id_to_fair_allocation[app.app_id]} demand: {app.demand}")
        '''

        residual = self._fractional_share
        app_id_to_allocation = {}

        if len(self._active_apps) == 0:
            return app_id_to_allocation

        for app_id in self._app_id_to_int_allocation:
            app_id_to_allocation[app_id] = int(self._app_id_to_int_allocation[app_id])

        # redistribute residual allocation based on round-robin

        temp_active_apps = [app.app_id for app in self._active_apps]

        hol_app, *temp_active_apps = temp_active_apps

        # random.shuffle(temp_active_apps)

        temp_active_apps = [hol_app] + temp_active_apps

        total_remaining_demand = sum([app.demand - app_id_to_allocation[app.app_id] for app in self._active_apps])

        # while total_remaining_demand > 0 and residual > 0:

        for app_id in temp_active_apps:


            # for app in self._active_apps:
            
            app = self._app_list[app_id]

            remaining_demand = app.demand - app_id_to_allocation[app.app_id]
            additional_allocation = min(residual, remaining_demand, 1)

            app_id_to_allocation[app.app_id] += additional_allocation
            residual -= additional_allocation
            total_remaining_demand -= additional_allocation

        assert(np.isclose(residual, 0)), residual

        return app_id_to_allocation

    def compute_fair_allocation(self):



        app_id_to_allocation = {}

        residual = int(self._max_capacity)

        app_demands = [0] * len(self._active_apps)
        app_remaining_allocation = [0] * len(self._active_apps)


        for i, app in enumerate(self._active_apps):
            app_id_to_allocation[app.app_id] = 0

            try:
                app_remaining_allocation[i] = frac(residual, len(app_demands))
            except Exception as e:
            
                print(residual)
                print(len(app_demands))

                raise e
            
            # app_remaining_allocation[i] = (residual/len(app_demands))
            app_demands[i] = app.demand


        total_demand = sum(app_demands)
        

        tries = 0
        while residual > 0 and total_demand > 0:

            for i, app in enumerate(self._active_apps):

                allocation = min(app_remaining_allocation[i], app_demands[i])

                app_id_to_allocation[app.app_id] += allocation

                app_demands[i] -= allocation
                
                residual -= allocation
                total_demand -= allocation

                if app_demands[i] == 0:
                    app_remaining_allocation[i] = 0


            if np.isclose(float(residual), 0.0) or np.isclose(float(total_demand), 0.0):
                break


            R = sum(app_remaining_allocation)
            if R > 0:
                # app_remaining_allocation = [residual * (allocation/R) for allocation in app_remaining_allocation]
                app_remaining_allocation = [residual * frac(allocation, R) for allocation in app_remaining_allocation]

            tries += 1
            if tries > 100:
                print(f"tries: {tries}")
                sys.exit(1)

        return app_id_to_allocation

    def run(self):


        if self._app_info_fn:
            with open(self._app_info_fn,'w') as fp:
                fp.write("app_id,submit_time,start_time,end_time,estimated_start_time,estimated_end_time,fair_act,service,num_apps_seen_diff\n")

        @ray.remote
        def gen_background_event(event_queue, event, sleep_time):
            sleep(sleep_time)
            event.event_time = datetime.now()
            event_queue.put(event)
            

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

        

            # remove failed apps
            if (datetime.now() - last_self_check_time).total_seconds() > self._inactivity_time:
                
                if len(self._event_queue) <= 5:
                    self.remove_failed_apps()
                    last_self_check_time = datetime.now()

            if resource_change_event or redivision_event:
                
                # recompute fair allocation
                if resource_change_event:
                    self._app_id_to_fair_allocation = {}
                    self._app_id_to_fractional_allocation = {}
                    self._app_id_to_int_allocation = {}
                    self._fractional_share = None

                    self._app_id_to_fair_allocation = self.compute_fair_allocation()    
                    
                    for app_id in self._app_id_to_fair_allocation:
                        self._app_id_to_fractional_allocation[app_id] = self._app_id_to_fair_allocation[app_id] - int(self._app_id_to_fair_allocation[app_id])
                        self._app_id_to_int_allocation[app_id] = int(self._app_id_to_fair_allocation[app_id])

                    self._fractional_share = int(sum([self._app_id_to_fractional_allocation[app_id] for app_id in self._app_id_to_fractional_allocation]))
                    self._quantum = self._min_quantum * len(self._active_apps)

                # left shift
                if len(self._active_apps) > 1 and not np.isclose(self._fractional_share, 0.0):
                    
                    self._active_apps.append(self._active_apps.pop(0))

                    next_app = self._active_apps[0]

                    next_redivision = (self._quantum * float(self._app_id_to_fractional_allocation[next_app.app_id]))

                    assert(float(next_redivision) >= 0)

                    gen_background_event.remote(self._event_queue,
                                            Event(event_id=None, event_type="REDIVISION", event_time=None),
                                            float(next_redivision))


            self.update_allocations(event.event_time)

            self.report_progress()
            
            
