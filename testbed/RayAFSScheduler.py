import ray
from RayGenericScheduler import RayAppGenericScheduler
import os, sys
from datetime import datetime, timedelta
from time import sleep
import math
from functools import partial
import copy


from AFSScheduler import AppAFSScheduler
from common import Event, App, Job


EPS=1e-7

class RayAppAFSScheduler(RayAppGenericScheduler):
    """docstring for RayAppMCScheduler"""
    def __init__(self, total_gpus, event_queue, app_list, app_info_fn="results.csv", suppress_print=False, estimate=True):
        super(RayAppAFSScheduler, self).__init__(total_gpus, event_queue, app_list, app_info_fn, suppress_print=suppress_print, estimate=estimate)
        print("Warning. compute_remaining_time makes 1job or linear scaling assumption")

    def sim_estimate(self, app):


        # total_gpus, event_queue, app_list, class_detail, app_info_fn="results.csv", suppress_print=False, estimate=False

        snap_shot = AppAFSScheduler(total_gpus=self._max_capacity,
                                    event_queue=[Event(event_id=app.app_id, event_time=datetime.now(), event_type=Event.APP_SUB, app_id=app.app_id)],
                                    app_list={},
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


    def alg_c_concept(self, total_gpus):

        js = list()
        
        app_id_to_allocation = {}

        for a in self._active_apps:
            a.tmp_gpus = 0

            if a.demand > 0:
                js.append(a)

        gpus = total_gpus

        while gpus > 0 and len(js) > 0:
            cand = js[0]
            for m in js[1:]:
                if self.compute_remaining_time(m, m.tmp_gpus) < self.compute_remaining_time(cand, cand.tmp_gpus):
                    l, h = m, cand
                else:
                    l, h = cand, m


                sl0 = 1 / (self.compute_remaining_time(l, l.tmp_gpus) + EPS)
                sl1 = 1 / (self.compute_remaining_time(l, l.tmp_gpus+1)+ EPS)
                sh0 = 1 / (self.compute_remaining_time(h, h.tmp_gpus)+ EPS)
                sh1 = 1 / (self.compute_remaining_time(h, h.tmp_gpus+1)+ EPS)


                if sl0 == 0:
                    cand = l if sl1 > sh1 else h
                elif (sl1 - sl0) / sl0 > (sh1 - sh0) / sh1:
                    cand = l
                else:
                    cand = h

            allocation_increment = min(1, cand.demand)

            cand.tmp_gpus += allocation_increment

            if cand.tmp_gpus == cand.demand:
                js.remove(cand)

            gpus -= allocation_increment
        


        for a in self._active_apps:
            app_id_to_allocation[a.app_id] = a.tmp_gpus
        return app_id_to_allocation



    def compute_remaining_time(self, app, app_current_allocation):

        if len(app.jobs) == 1:
            thrpt = app.jobs[0].thrpt(app_current_allocation)
        else:
            thrpt = min(app.demand, app_current_allocation)

        if thrpt > 0:        
            return app.remaining_service/thrpt
        else:
            pass
            # print(f"app_id: {app.app_id} thrpt: {thrpt} app_current_allocation: {app_current_allocation} app.demand: {app.demand} remaining_service: {app.remaining_service} job_statuses: {[j.status for j in app.jobs.values()]} app_status: {app.status}")

        return float('inf')

    def compute_allocation(self, event_time):
        
        
        total_demand = sum([app.demand for app in self._active_apps])

        residual = min(total_demand, self._max_capacity)


        app_id_to_allocation = self.alg_c_concept(self._max_capacity)

        print(f"DEBUG: total_demand: {total_demand} residual: {residual} app_id_to_allocation: {app_id_to_allocation}")

    
        for app_id in app_id_to_allocation:
            
            allocation = app_id_to_allocation[app_id]

            residual -= allocation
            demand = self._app_list[app_id].demand

            assert(allocation <= demand), (allocation, demand, app_id)


        assert(math.isclose(residual, 0, abs_tol=1e-3)), residual


        return app_id_to_allocation