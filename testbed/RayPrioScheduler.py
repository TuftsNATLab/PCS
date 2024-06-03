import ray
from RayGenericScheduler import RayAppGenericScheduler
import os, sys
from datetime import datetime, timedelta
from time import sleep
import numpy as np
from functools import partial
import copy

from PriorityScheduler import AppPrioScheduler
from common import Event, App, Job

class RayAppPrioScheduler(RayAppGenericScheduler):

    """docstring for AppPrioScheduler"""
    def __init__(self, total_gpus, event_queue, app_list, prio_func, app_info_fn="results.csv", suppress_print=False, estimate=True):
        super(RayAppPrioScheduler, self).__init__(total_gpus, event_queue, app_list, app_info_fn, suppress_print=suppress_print, estimate=estimate)
        self._prio_func = prio_func
        


    def sim_estimate(self, app):


        snap_shot = AppPrioScheduler(total_gpus=self._max_capacity,
                                    event_queue=[Event(event_id=app.app_id, event_time=datetime.now(), event_type=Event.APP_SUB, app_id=app.app_id)],
                                    app_list={},
                                    prio_func=self._prio_func,
                                    app_info_fn=None)


        snap_shot._active_apps = copy.deepcopy(self._active_apps)
        snap_shot._app_id_to_allocation = copy.deepcopy(self._app_id_to_allocation)

        #TODO: populate end_event_list - issue, I don't know job allocation

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


        assert(np.isclose(allocation, 0)), allocation

        return app_id_to_allocation