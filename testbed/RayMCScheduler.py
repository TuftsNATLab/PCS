import ray
from RayGenericScheduler import RayAppGenericScheduler
import os, sys
from datetime import datetime, timedelta
from time import sleep
import numpy as np
from functools import partial
import copy
from fractions import Fraction as frac


from MCSScheduler import AppPracticalMCScheduler
from common import Event, App, Job


class RayAppMCScheduler(RayAppGenericScheduler):
    """docstring for RayAppMCScheduler"""
    def __init__(self, total_gpus, event_queue, app_list, class_detail, quantum=100, app_info_fn="results.csv", suppress_print=False, estimate=True):
        super(RayAppMCScheduler, self).__init__(total_gpus, event_queue, app_list, app_info_fn, suppress_print=suppress_print, estimate=estimate)
    

        self._class_detail = copy.deepcopy(class_detail)
        self._num_classes = class_detail["num_classes"]
        self._class_thresholds = class_detail["class_thresholds"]
        self._default_class_rates = copy.copy(class_detail["class_rates"])
        self._clip_demand_factor = class_detail["clip_demand_factor"]
        self._delta = class_detail["delta"]
        assert(np.isclose(float(sum(self._default_class_rates)), 1.0)), float(sum(self._default_class_rates))
        
        for i in range(self._num_classes):
            self._default_class_rates[i] = self._default_class_rates[i] * self._max_capacity

        
        self._class_rates = self._default_class_rates[:]


        self._quantum = quantum
        self._app_id_to_fair_allocation = {}
        self._app_id_to_fractional_allocation = {}
        self._app_id_to_int_allocation = {}
        self._fractional_share = None
        self._sharing_group = list()
    
    def sim_estimate(self, app):

        snap_shot = AppPracticalMCScheduler(total_gpus=self._max_capacity,
                                    event_queue=[Event(event_id=app.app_id, event_time=datetime.now(), event_type=Event.APP_SUB, app_id=app.app_id)],
                                    app_list={},
                                    quantum=self._quantum,
                                    class_detail=self._class_detail,
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

        try:
            snap_shot.run(partial(break_cond, snap_shot._app_list[app.app_id]))
        except Exception as e:
            raise e


        

        app.update_estimates(snap_shot._app_list[app.app_id].start_time,
                            snap_shot._app_list[app.app_id].end_time)

    def classifier(self, app):
        for i, t in enumerate(self._class_thresholds):
            if app.service <= t:
                return i
        return i



    def __clip_demand(self, app, clip_demand_factor=0.9):
        

        for job in app.jobs.values():
            job.optimal_demand = job.demand
            for g in range(1,job.demand+1):
                if job.thrpt_dic[g]/float(g) < clip_demand_factor:
                    job.optimal_demand = g-1
                    break
        
        app.optimal_demand = sum([job.optimal_demand if job.status != Job.END else 0 for job in app.jobs.values()])


    def __intra_class_allocation(self, app_class, class_allocation, app_id_to_allocation):
        class_apps = list(filter(lambda a: a.app_class == app_class, self._active_apps))
        

        # sort by who came first
        class_apps = sorted(class_apps, key=lambda a: a.app_id)


        delta=self._delta
        clip_demand_factor = self._clip_demand_factor

        starting_class_allocation = class_allocation

        for app in class_apps:
            app_id_to_allocation[app.app_id] = 0

        while class_allocation > 0.0:
            class_allocation = starting_class_allocation
            for app in class_apps:
                self.__clip_demand(app, clip_demand_factor)
                app_id_to_allocation[app.app_id] = float(min(class_allocation, app.optimal_demand))
                class_allocation -= app_id_to_allocation[app.app_id]

            clip_demand_factor -= delta


        for app in class_apps:
            assert(app.demand >= app_id_to_allocation[app.app_id]) 
        assert(np.isclose(float((class_allocation)), 0.0)), class_allocation


    def compute_MCS_allocation(self, event_time):

        class_demand = [0] * self._num_classes                
        class_allocation = [0] * self._num_classes
        self._class_rates = self._default_class_rates[:]

        for app in self._active_apps:
            class_demand[app.app_class] += app.demand


        total_demand = sum(class_demand)
        residual = sum(self._default_class_rates)


        tries = 0

        while residual > 0 and total_demand > 0:
            for i in range(self._num_classes):

                allocation = min(self._class_rates[i], class_demand[i])

                if np.isclose(float(allocation), 0):
                    allocation = 0

                class_allocation[i] += allocation
                class_demand[i] -= allocation
                residual -= allocation
                total_demand -= allocation

                if np.isclose(float(class_demand[i]), 0):
                    self._class_rates[i] = 0
                    class_demand[i] = 0


            if np.isclose(float(residual), 0.0) or np.isclose(float(total_demand), 0.0):
                break


            R = sum(self._class_rates)
            if R > 0:
                self._class_rates = [frac(residual * self._class_rates[i], R) for i in range(self._num_classes)]

            tries += 1

            if tries > 100:
                break
                # raise Exception("Too many while loops")

        # after this while loop, we have gpu allocations per class in the class_allocation vector


        self._class_rates = class_allocation[:]
        app_id_to_allocation = {}

        for app_class in range(self._num_classes):
            self.__intra_class_allocation(app_class, class_allocation[app_class], app_id_to_allocation)

        return app_id_to_allocation



    def compute_allocation(self, event_time):


        app_id_to_mcs_allocation = self.compute_MCS_allocation(event_time)

        residual = self._fractional_share
        app_id_to_allocation = {}


        # assign int allocation
        for app_id in self._app_id_to_int_allocation:
            app_id_to_allocation[app_id] = int(self._app_id_to_int_allocation[app_id])


        # these apps are those with fractional share
        total_remaining_demand = sum([app.demand - app_id_to_allocation[app.app_id] for app in self._sharing_group])

        for app in self._sharing_group:

            remaining_demand = app.demand - app_id_to_allocation[app.app_id]
            additional_allocation = min(residual, remaining_demand, 1)

            app_id_to_allocation[app.app_id] += additional_allocation
            residual -= additional_allocation
            total_remaining_demand -= additional_allocation

        return app_id_to_allocation

    def handle_app_sub_event(self, event):
        super(RayAppMCScheduler, self).handle_app_sub_event(event)
        app = self._app_list[event.app_id]
        app.app_class = self.classifier(app)



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
                    self._app_id_to_MCS_allocation = {}
                    self._app_id_to_fractional_allocation = {}
                    self._app_id_to_int_allocation = {}
                    self._fractional_share = None
                    self._sharing_group = list()

                    self._app_id_to_MCS_allocation = self.compute_MCS_allocation(event.event_time)



                    for app_id in self._app_id_to_MCS_allocation:
                        self._app_id_to_fractional_allocation[app_id] = self._app_id_to_MCS_allocation[app_id] - int(self._app_id_to_MCS_allocation[app_id])

                        self._app_id_to_int_allocation[app_id] = int(self._app_id_to_MCS_allocation[app_id])

                        if self._app_id_to_fractional_allocation[app_id] > 0:
                            self._sharing_group.append(self._app_list[app_id])

                    self._fractional_share = int(sum([self._app_id_to_fractional_allocation[app_id] for app_id in self._app_id_to_fractional_allocation]))
                    
                    # self._quantum = self._min_quantum * len(self._active_apps)

                # left shift
                if len(self._sharing_group) > 1 and not np.isclose(self._fractional_share, 0.0):
                    
                    self._sharing_group.append(self._sharing_group.pop(0))

                    next_app = self._sharing_group[0]

                    next_redivision = (self._quantum * float(self._app_id_to_fractional_allocation[next_app.app_id]))

                    assert(float(next_redivision) >= 0)

                    gen_background_event.remote(self._event_queue,
                                            Event(event_id=None, event_type="REDIVISION", event_time=None),
                                            float(next_redivision))


            self.update_allocations(event.event_time)
            self.report_progress()