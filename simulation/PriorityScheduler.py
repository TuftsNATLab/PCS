import sys

import copy
import math
from datetime import datetime

from common import Event, App, Job
from GenericScheduler import AppGenericScheduler

class AppPrioScheduler(AppGenericScheduler):
    """docstring for AppPrioScheduler"""
    def __init__(self, total_gpus, event_queue, app_list, prio_func, app_info_fn="results.csv", suppress_print=False):
        super(AppPrioScheduler, self).__init__(total_gpus, event_queue, app_list, app_info_fn, suppress_print)
        self._prio_func = prio_func
        

    def compute_allocation(self, event_time):
        

        '''
        for app in self._active_apps:

            if app.demand == 0:
                print(f"app_id: {app.app_id}")
                for job in app.jobs.values():
                    print(f"job_id: {job.job_id} status: {job.status} demand: {job.demand} remaining_service: {job.remaining_service}")

                assert(False)
        '''




        self._active_apps = sorted(self._active_apps, key=lambda t: self._prio_func(t))

        total_demand = sum([app.demand for app in self._active_apps])

        allocation = min(total_demand, self._max_capacity)


        app_id_to_allocation = {}

        for app in self._active_apps:

                
            app_id_to_allocation[app.app_id] = min(allocation, app.demand)
            allocation -= app_id_to_allocation[app.app_id]


        assert(math.isclose(allocation, 0)), allocation

        return app_id_to_allocation