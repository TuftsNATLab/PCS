import sys

import copy
import numpy as np

import math

import warnings

from GenericScheduler import AppGenericScheduler
from common import Event, App, Job


EPS=1e-7

class AppAFSScheduler(AppGenericScheduler):
    """docstring for AppAFSScheduler"""
    def __init__(self, total_gpus, event_queue, app_list, app_info_fn="results.csv", suppress_print=False, verbosity=1, p_error=None):
        super(AppAFSScheduler, self).__init__(total_gpus, event_queue, app_list, app_info_fn, suppress_print, verbosity, p_error)
        print("Warning. compute_remaining_time makes 1job or linear scaling assumption")
                

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
            return app.estimated_remaining_service/thrpt

        return float('inf')

    def compute_allocation(self, event_time):
    

        total_demand = sum([app.demand for app in self._active_apps])

        residual = min(total_demand, self._max_capacity)


        app_id_to_allocation = self.alg_c_concept(self._max_capacity)

        for app_id in app_id_to_allocation:
            
            allocation = app_id_to_allocation[app_id]

            residual -= allocation
            demand = self._app_list[app_id].demand

            assert(allocation <= demand), (allocation, demand, app_id)


        assert(math.isclose(residual, 0, abs_tol=1e-3)), residual

        return app_id_to_allocation