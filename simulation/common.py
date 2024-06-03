import copy
from datetime import datetime, timedelta
import numpy as np
import math
from functools import wraps


def memoize(func):
    memo = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key not in memo:
            memo[key] = func(*args, **kwargs)
        return memo[key]

    return wrapper

class App(object):

    UNDEFINED = "UNDEFINED"
    SUBMITTED = "SUBMITTED"
    ACTIVE = "ACTIVE"
    QUEUED = "QUEUED"
    PREEMPTED = "PREEMPTED"
    FAILED = "FAILED"
    END = "END"
    PENDING_END = "PENDING_END"

    
    """docstring for App"""
    def __init__(self, app_id=0, jobs={}, deadline=None):
        super(App, self).__init__()
        self.app_id = app_id
        self.jobs = copy.deepcopy(jobs)
        self.deadline = deadline


        self.submit_time = datetime.max
        self.start_time = datetime.max
        self.end_time = datetime.max


        self.status = App.UNDEFINED
        self.estimated_start_time = []
        self.estimated_end_time = []
            
        self.allocation = 0

        # debatable
        self.app_class = 0
        self.prio = 0
        self.num_apps_seen = None
        self.service = sum([job.remaining_service for job in self.jobs.values()])

        self.demand = sum([job.demand if job.status != Job.END else 0 for job in self.jobs.values()])
        self.remaining_service = self.service

        self.initial_demand = self.demand
        self._context = {}

        # new addons
        self.estimated_service = self.service
        self.estimated_remaining_service = sum([job.estimated_remaining_service for job in self.jobs.values()])
        self.induced_error = 0

    
    def __deepcopy__(self, memo):

        new_instance = self.__class__(self.app_id, {}, self.deadline)
        new_instance.submit_time = self.submit_time
        new_instance.start_time = self.start_time
        new_instance.end_time = self.end_time

        new_instance.allocation = self.allocation
        new_instance.num_apps_seen = self.num_apps_seen

        new_instance.status = self.status
        new_instance.app_class = self.app_class
        new_instance.demand = self.demand
        new_instance.service = self.estimated_service
        new_instance.remaining_service = self.estimated_remaining_service
        new_instance.estimated_service = self.estimated_service
        new_instance.estimated_remaining_service = self.estimated_remaining_service
        new_instance.initial_demand = self.initial_demand
        new_instance.induced_error = self.induced_error
        new_instance.jobs = copy.deepcopy(self.jobs)
        return new_instance


    @property
    def min_demand(self):
        min_demand_list = [job.min_demand for job in self.jobs.values() if job.status != Job.END]
        if len(min_demand_list) > 0:
            return min(min_demand_list)
        return 0
    
    def update_allocation(self, allocation):
        residual = allocation

        '''
        print(f"DEBUG: self.status: {self.status}")
        for job in self.jobs.values():
            print(f"DEBUG: job.status: {job.status}")
        '''

        for job in self.jobs.values():
            if job.status != Job.END:

                if math.isclose(residual, 0):
                    residual = 0

                job.allocation = min(job.demand, residual)

                # print(f"DEBUG: {job.demand}, {residual}")


                residual -= job.allocation
            else:
                job.allocation = 0
        self.allocation = allocation - residual
        return residual


    def update_estimates(self, estimated_start_time, estimated_end_time):
        self.estimated_start_time.append(estimated_start_time)
        self.estimated_end_time.append(estimated_end_time)


    def on_app_submit(self, event_time):
        for job in self.jobs.values():
            job.status = Job.SUBMITTED
            job.submit_time = event_time

    def on_app_start(self, event_time):
        for job in self.jobs.values():
            
            if job.status == Job.END:
                continue

            '''
            if job.allocation > 0:
                # print(event_time + timedelta(seconds=job.remaining_service/job.allocation))
                # print(timedelta(seconds=job.remaining_service/job.allocation))
                
                assert(event_time != datetime.max)

                try:
                    job.attempts.append({"start_time": event_time, "end_time": event_time + timedelta(seconds=job.remaining_service/job.allocation)})
                except Exception as e:
                    
                    print(f"job.remaining_service: {job.remaining_service}")
                    print(f"job.allocation: {job.allocation}")

                    raise e

                
                
                
            else:
                job.attempts.append({"start_time": event_time, "end_time": datetime.max})
            '''
    
            job.status = Job.ACTIVE

    def exec_func(self, event_queue):
        pass

    def on_job_end(self, job_id, event_time):
        job = self.jobs[job_id]
        self.demand -= job.demand
        job.on_job_end(event_time)



    # def __gt__(self, other):
    #     return self.prio > other.prio
    # def __lt__(self, other):
    #     return self.prio < other.prio
    # def __eq__(self, other):
    #     return self.prio == other.prio

class Job(object):


    UNDEFINED = "UNDEFINED"
    SUBMITTED = "SUBMITTED"
    ACTIVE = "ACTIVE"
    QUEUED = "QUEUED"
    PREEMPTED = "PREEMPTED"
    FAILED = "FAILED"
    END = "END"



    """docstring for Job"""
    def __init__(self, app_id=0, job_id=0,  service=0, demand=0, min_demand=0):
        super(Job, self).__init__()
        self.app_id = app_id
        self.job_id = job_id
        self.service = service
        
        self.demand = demand
        self.min_demand = min_demand


        self.submit_time = datetime.max
        self.start_time = datetime.max
        self.end_time = datetime.max

        self.status = Job.UNDEFINED
        self.attempts = list()
        
        self.remaining_service = service

        self.estimated_service = service
        self.estimated_remaining_service = service

        self.allocation = 0
        self.last_event_time = datetime.max

        self.thrpt_dic = None

    
    def __deepcopy__(self, memo):

        new_instance = self.__class__(self.app_id, self.job_id, self.estimated_remaining_service, self.demand, self.min_demand)


        new_instance.submit_time = self.submit_time
        new_instance.start_time = self.start_time
        new_instance.end_time = self.end_time

        new_instance.status = self.status
        
        new_instance.last_event_time = self.last_event_time

        new_instance.thrpt_dic = self.thrpt_dic

        new_instance.allocation = self.allocation

        return new_instance
    
        

    def on_job_end(self, event_time):
        self.status = Job.END
        self.end_time = event_time
        self.demand = 0

    # @memoize
    def thrpt(self, alloc):


        if alloc > self.demand:
            return self.thrpt_dic[self.demand]


        idx_c = int(math.ceil(alloc))
        idx_f = int(math.floor(alloc))

        m = self.thrpt_dic[idx_c] - self.thrpt_dic[idx_f]
        c = self.thrpt_dic[idx_c] - (m*idx_c)

        return m*alloc + c

    
    # def __gt__(self, other):
    #     return self.prio > other.prio
    # def __lt__(self, other):
    #     return self.prio < other.prio
    # def __eq__(self, other):
    #     return self.prio == other.prio
    
class Event(object):


    APP_SUB = "APP_SUB"
    APP_START = "APP_START"
    APP_END = "APP_END"
    JOB_SUB = "JOB_SUB"
    JOB_START = "JOB_START"
    JOB_END = "JOB_END"
    UNDEFINED = "UNDEFINED"


    """docstring for Event"""
    def __init__(self, event_id, event_time, event_type, **kwargs):
        super(Event, self).__init__()
        self.event_id = event_id
        self.event_time = event_time
        self.event_type = event_type

        for key in kwargs:
            setattr(self, key, kwargs[key])



    def __repr__(self):
        return f"Event: id: {self.event_id}, time: {self.event_time}, event_type: {self.event_type}"

    
    def __criteria(self):
        return self.event_time

    def __gt__(self, other):
        if self.__criteria() == other.__criteria():
            if "SUB" in self.event_type and "END" in other.event_type:
                return True
            elif "END" in self.event_type and "SUB" in other.event_type:
                return False
            else:
                return self.event_id > other.event_id
        return (self.__criteria() > other.__criteria())
    def __lt__(self, other):
        if self.__criteria() == other.__criteria():
            if "SUB" in self.event_type and "END" in other.event_type:
                return False
            elif "END" in self.event_type and "SUB" in other.event_type:
                return True
            else:
                return self.event_id < other.event_id
        return (self.__criteria() < other.__criteria())
    def __eq__(self, other):
        return (self.__criteria() == other.__criteria())