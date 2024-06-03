import sys, os

import copy, random
from datetime import datetime, timedelta
from heapq import heappush, heappop, heapify
from fractions import Fraction as frac
import math


from common import Event, App, Job
from GenericScheduler import AppGenericScheduler


class AppFairScheduler(AppGenericScheduler):
    """This class implements the Fair Scheduler for Apps"""
    # Non work conserving
    def __init__(self, total_gpus, event_queue, app_list, app_info_fn="results.csv", suppress_print=False, estimate=False):
        super(AppFairScheduler, self).__init__(total_gpus, event_queue, app_list, app_info_fn, suppress_print, estimate)


    def compute_allocation(self, event_time):


        app_id_to_allocation = {}

        residual = self._max_capacity

        app_demands = [0] * len(self._active_apps)
        app_remaining_allocation = [0] * len(self._active_apps)


        for i, app in enumerate(self._active_apps):
            app_id_to_allocation[app.app_id] = 0
            app_remaining_allocation[i] = residual/len(app_demands)
            app_demands[i] = app.demand


        total_demand = sum(app_demands)
        

        tries = 0
        while (not math.isclose(residual, 0.0, abs_tol=1e-6)) and (not math.isclose(total_demand, 0.0, abs_tol=1e-6)):

            for i, app in enumerate(self._active_apps):

                allocation = min(app_remaining_allocation[i], app_demands[i])

                app_id_to_allocation[app.app_id] += allocation
                app_demands[i] -= allocation
                
                residual -= allocation
                total_demand -= allocation

                if app_demands[i] == 0:
                    app_remaining_allocation[i] = 0


            if math.isclose(residual, 0.0, abs_tol=1e-6) or math.isclose(total_demand, 0.0, abs_tol=1e-6):
                break


            R = sum(app_remaining_allocation)
            if R > 0:
                app_remaining_allocation = [residual * allocation/R for allocation in app_remaining_allocation]

            tries += 1
            if tries > 100:
                print(f"tries: {tries} residual: {residual} total_demand: {total_demand} math.isclose(): {math.isclose(residual, 0.0, abs_tol=1e-6)}")
                sys.exit(1)


        fractional_share = [round(100.0*(app_id_to_allocation[app_id] - int(app_id_to_allocation[app_id]))) for app_id in app_id_to_allocation]

        

        return app_id_to_allocation


class AppPracticalFairScheduler(AppGenericScheduler):
    """This class implements a Practical Fair Scheduler for Apps"""
    # Non work conserving
    def __init__(self, total_gpus, event_queue, app_list, quantum=600, app_info_fn="results.csv", suppress_print=False, estimate=False):
        super(AppPracticalFairScheduler, self).__init__(total_gpus, event_queue, app_list, app_info_fn, suppress_print, estimate)

        # print("Init App Fair Scheduler with %d GPUs" % total_gpus)

        self._redivision_event_queue = list()
        self._quantum = quantum
        self._app_id_to_fair_allocation = {}
        self._app_id_to_fractional_allocation = {}
        self._app_id_to_int_allocation = {}
        self._fractional_share = None
        self._last_redivision=None


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

        assert(math.isclose(residual, 0)), residual

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


            if math.isclose(float(residual), 0.0) or math.isclose(float(total_demand), 0.0):
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
        total_allocation = self._fractional_share

        # left shift
        if len(self._active_apps) > 1:
            
            self._active_apps.append(self._active_apps.pop(0))

            if not math.isclose(total_allocation, 0.0):

                next_app = self._active_apps[0]

                
                next_redivision = (self._quantum * float(self._app_id_to_fractional_allocation[next_app.app_id]))

                assert(float(next_redivision) >= 0)

                # for app in self._active_apps:
                #     print(f"app_id: {app.app_id} fair_share: {self._app_id_to_fair_allocation[app.app_id]} frac_share: {self._app_id_to_fractional_allocation[app.app_id]}")

                self._redivision_event_queue = list()
                # heappush(self._redivision_event_queue, Event(event_id=0, event_time=event.event_time + timedelta(seconds=float(next_redivision)), event_type="REDIVISION"))
                heappush(self._redivision_event_queue, Event(event_id=0, event_time=event.event_time + timedelta(seconds=float(next_redivision)), event_type="REDIVISION"))




    def run(self):

        while len(self._event_queue) > 0 or len(self._end_event_queue) > 0:

            event = heappop(self.__pick_min_heap(self.__pick_min_heap(self._event_queue, self._end_event_queue), self._redivision_event_queue))

            self.progress_active_apps(event.event_time)            
            self._last_event_time = event.event_time



            self.report_progress(event)

            if event.event_type == Event.APP_SUB:
                self.handle_app_sub_event(event)

            elif event.event_type == Event.JOB_END:
                self.handle_job_end_event(event)


            if event.event_type in [Event.APP_SUB, Event.JOB_END]:
                

                self._app_id_to_fair_allocation = {}
                self._app_id_to_fractional_allocation = {}
                self._app_id_to_int_allocation = {}
                self._fractional_share = None

                self._app_id_to_fair_allocation = self.compute_fair_allocation()    
                

                '''
                print(f"event_time: {self.absolute_time(event.event_time)}")
                for app_id in self._app_id_to_fair_allocation:
                    print(f"app_id: {app_id} fair_allocation: {self._app_id_to_fair_allocation[app_id]}")
                print("===========================")
                '''


                for app_id in self._app_id_to_fair_allocation:
                    self._app_id_to_fractional_allocation[app_id] = self._app_id_to_fair_allocation[app_id] - int(self._app_id_to_fair_allocation[app_id])
                    self._app_id_to_int_allocation[app_id] = int(self._app_id_to_fair_allocation[app_id])


                self._fractional_share = int(sum([self._app_id_to_fractional_allocation[app_id] for app_id in self._app_id_to_fractional_allocation]))

                self.rounds=0
            if event.event_type in [Event.APP_SUB, Event.JOB_END, "REDIVISION"]:

                if self.rounds == 0:
                    self.rounds = len(self._active_apps)
                    self._last_redivision = event.event_time

                    

                else:

                    self.rounds -= 1
                    self._last_redivision = event.event_time

                self.redivision(event)

            self.update_allocations(event.event_time)

            self.update_end_events(event.event_time)


            '''
            if len(self._active_apps) == 3:
                print(f"{self._active_apps[0].app_id},{self._active_apps[1].app_id},{self._active_apps[2].app_id}"+\
                    f"->{self._active_apps[0].allocation},{self._active_apps[1].allocation},{self._active_apps[2].allocation}")
            '''


            # self.util_print_progress(event)


            if event.event_type == Event.APP_SUB and self._estimate:
                self.sim_estimate(app = self._app_list[event.app_id])