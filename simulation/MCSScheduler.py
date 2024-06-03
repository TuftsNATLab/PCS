import os
import sys
import copy
from datetime import datetime, timedelta
import math
import random
import ray
from GenericScheduler import AppGenericScheduler
from GenericScheduler import multi_runner
from common import Event, App, Job
from fractions import Fraction as frac


class AppMCScheduler(AppGenericScheduler):
    """This class implements a Multi-Class Scheduler with fixed rate for Apps"""

    def __init__(
        self,
        total_gpus,
        event_queue,
        app_list,
        class_detail,
        app_info_fn="results.csv",
        suppress_print=False,
        verbosity=1,
        p_error=None,
    ):
        super(AppMCScheduler, self).__init__(
            total_gpus,
            event_queue,
            app_list,
            app_info_fn,
            suppress_print,
            verbosity=verbosity,
            p_error=p_error,
        )

        self._class_detail = copy.deepcopy(class_detail)
        self._num_classes = class_detail["num_classes"]
        self._class_thresholds = class_detail["class_thresholds"]
        self._default_class_rates = copy.copy(class_detail["class_rates"])
        self._clip_demand_factor = class_detail["clip_demand_factor"]
        self._delta = class_detail["delta"]

        assert math.isclose(float(sum(self._default_class_rates)), 1.0, abs_tol=1e-3), float(
            sum(self._default_class_rates)
        )

        for i in range(self._num_classes):
            self._default_class_rates[i] = (
                self._default_class_rates[i] * self._max_capacity
            )

        self._class_rates = self._default_class_rates[:]

    def __average_service(self, app):
        app_service = app.service

        s = 0
        for i in range(1, len(app.jobs[0].thrpt_dic)):
            s += i / app.jobs[0].thrpt_dic[i]

        return (app_service / (i - 1)) * s

    # or app_service = class_rate * app.service/thrpt(class_rate)

    def __clip_demand(self, app, clip_demand_factor=0.9):
        for job in app.jobs.values():
            job.optimal_demand = job.demand
            for g in range(1, job.demand + 1):
                if job.thrpt_dic[g] / float(g) < clip_demand_factor:
                    job.optimal_demand = g - 1
                    break

        app.optimal_demand = sum(
            [
                job.optimal_demand if job.status != Job.END else 0
                for job in app.jobs.values()
            ]
        )

    def classifier(self, app, induce_error=False):
        app_service = app.estimated_service

        for i, t in enumerate(self._class_thresholds):
            if app_service <= t:
                return i
        return i

    def handle_app_sub_event(self, event):
        super(AppMCScheduler, self).handle_app_sub_event(event)
        app = self._app_list[event.app_id]
        app.app_class = self.classifier(app)
        # self.__clip_demand(app)

    def compute_allocation_non_work_serving(self, event_time):
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

                if math.isclose(float(allocation), 0, abs_tol=1e-3):
                    allocation = 0

                class_allocation[i] += allocation
                class_demand[i] -= allocation
                residual -= allocation
                total_demand -= allocation

                if math.isclose(float(class_demand[i]), 0, abs_tol=1e-3):
                    self._class_rates[i] = 0
                    class_demand[i] = 0

            if math.isclose(float(residual), 0.0, abs_tol=1e-3) or math.isclose(
                float(total_demand), 0.0, abs_tol=1e-3
            ):
                break

            R = sum(self._class_rates)
            if R > 0:
                self._class_rates = [
                    frac(residual * self._class_rates[i], R)
                    for i in range(self._num_classes)
                ]

            tries += 1

            if tries > 100:
                break
                # raise Exception("Too many while loops")

        self._class_rates = class_allocation[:]
        app_id_to_allocation = {}

        for app in self._active_apps:
            # app_id_to_allocation[app.app_id] = min(class_allocation[app.app_class] if class_allocation[app.app_class] >= app.min_demand else 0, app.demand)
            app_id_to_allocation[app.app_id] = float(
                min(class_allocation[app.app_class], app.demand)
            )
            class_allocation[app.app_class] -= app_id_to_allocation[app.app_id]

        assert math.isclose(float(sum(class_allocation)), 0, abs_tol=1e-3), class_allocation

        return app_id_to_allocation

    def __intra_class_allocation_afs_style(
        self, app_class, class_allocation, app_id_to_allocation
    ):
        class_apps = list(filter(lambda a: a.app_class == app_class, self._active_apps))

        # allocate guaranteed
        for app in class_apps:
            app_id_to_allocation[app.app_id] = float(
                min(class_allocation, app.optimal_demand)
            )
            class_allocation -= app_id_to_allocation[app.app_id]

        # allocate leftover
        while class_allocation > 0.0:
            potential_allocation = min(1.0, class_allocation)

            class_apps = sorted(
                class_apps,
                key=lambda a: (
                    a.jobs[0].thrpt(
                        app_id_to_allocation[a.app_id] + potential_allocation
                    )
                    - a.jobs[0].thrpt(app_id_to_allocation[a.app_id]),
                    -1 * a.app_id,
                ),
            )

            optimal_app = class_apps[-1]
            app_id_to_allocation[optimal_app.app_id] += potential_allocation

            assert optimal_app.demand >= app_id_to_allocation[optimal_app.app_id]

            class_allocation -= potential_allocation

        assert math.isclose(float((class_allocation)), 0.0, abs_tol=1e-3), class_allocation

    def __intra_class_allocation(
        self, app_class, class_allocation, app_id_to_allocation
    ):
        class_apps = list(filter(lambda a: a.app_class == app_class, self._active_apps))

        # sort by who came first
        class_apps = sorted(class_apps, key=lambda a: a.app_id)

        delta = self._delta
        clip_demand_factor = self._clip_demand_factor

        starting_class_allocation = class_allocation

        while class_allocation > 0.0:
            class_allocation = starting_class_allocation
            for app in class_apps:
                self.__clip_demand(app, clip_demand_factor)
                app_id_to_allocation[app.app_id] = float(
                    min(class_allocation, app.optimal_demand)
                )
                class_allocation -= app_id_to_allocation[app.app_id]

            clip_demand_factor -= delta

        for app in class_apps:
            assert app.demand >= app_id_to_allocation[app.app_id]
        assert math.isclose(float((class_allocation)), 0.0, abs_tol=1e-3), class_allocation

    def compute_allocation(self, event_time):
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

                if math.isclose(float(allocation), 0, abs_tol=1e-3):
                    allocation = 0

                class_allocation[i] += allocation
                class_demand[i] -= allocation
                residual -= allocation
                total_demand -= allocation

                if math.isclose(float(class_demand[i]), 0, abs_tol=1e-3):
                    self._class_rates[i] = 0
                    class_demand[i] = 0

            if math.isclose(float(residual), 0.0, abs_tol=1e-3) or math.isclose(
                float(total_demand), 0.0, abs_tol=1e-3):
                break

            R = sum(self._class_rates)
            if R > 0:
                self._class_rates = [
                    frac(residual * self._class_rates[i], R)
                    for i in range(self._num_classes)
                ]

            tries += 1

            if tries > 100:
                break
                # raise Exception("Too many while loops")

        # after this while loop, we have gpu allocations per class in the class_allocation vector

        self._class_rates = class_allocation[:]
        app_id_to_allocation = {}

        for app_class in range(self._num_classes):
            self.__intra_class_allocation(
                app_class, class_allocation[app_class], app_id_to_allocation
            )

        return app_id_to_allocation


class AppPracticalMCScheduler(AppGenericScheduler):
    """docstring for AppPracticalMCScheduler"""

    def __init__(
        self,
        total_gpus,
        event_queue,
        app_list,
        class_detail,
        quantum=100,
        app_info_fn="results.csv",
        suppress_print=False,
        verbosity=1,
        p_error=None,
    ):
        super(AppPracticalMCScheduler, self).__init__(
            total_gpus,
            event_queue,
            app_list,
            app_info_fn,
            suppress_print,
            verbosity=verbosity,
            p_error=p_error,
        )

        self._class_detail = copy.deepcopy(class_detail)
        self._num_classes = class_detail["num_classes"]
        self._class_thresholds = class_detail["class_thresholds"]
        self._default_class_rates = copy.copy(class_detail["class_rates"])
        self._clip_demand_factor = class_detail["clip_demand_factor"]
        self._delta = class_detail["delta"]
        assert math.isclose(float(sum(self._default_class_rates)), 1.0, abs_tol=1e-3), float(
            sum(self._default_class_rates)
        )

        for i in range(self._num_classes):
            self._default_class_rates[i] = (
                self._default_class_rates[i] * self._max_capacity
            )

        self._class_rates = self._default_class_rates[:]

        self._redivision_event = None
        self._quantum = quantum
        self._app_id_to_fractional_allocation = {}
        self._app_id_to_int_allocation = {}
        self._fractional_share = None
        self._sharing_group = list()

    def __clip_demand(self, app, clip_demand_factor=0.9):
        for job in app.jobs.values():
            job.optimal_demand = job.demand
            for g in range(1, job.demand + 1):
                if job.thrpt_dic[g] / float(g) < clip_demand_factor:
                    job.optimal_demand = g - 1
                    break

        app.optimal_demand = sum(
            [
                job.optimal_demand if job.status != Job.END else 0
                for job in app.jobs.values()
            ]
        )

    def classifier(self, app, induce_error=False):
        app_service = app.estimated_service

        for i, t in enumerate(self._class_thresholds):
            if app_service <= t:
                return i
        return i

    def handle_app_sub_event(self, event):
        super(AppPracticalMCScheduler, self).handle_app_sub_event(event)
        app = self._app_list[event.app_id]
        app.app_class = self.classifier(app)
        # self.__clip_demand(app)

    def __intra_class_allocation(
        self, app_class, class_allocation, app_id_to_allocation
    ):
        class_apps = list(filter(lambda a: a.app_class == app_class, self._active_apps))

        # sort by who came first
        class_apps = sorted(class_apps, key=lambda a: a.app_id)

        delta = self._delta
        clip_demand_factor = self._clip_demand_factor

        starting_class_allocation = class_allocation

        while class_allocation > 0.0:
            class_allocation = starting_class_allocation
            for app in class_apps:
                self.__clip_demand(app, clip_demand_factor)
                app_id_to_allocation[app.app_id] = float(
                    min(class_allocation, app.optimal_demand)
                )
                class_allocation -= app_id_to_allocation[app.app_id]

            clip_demand_factor -= delta

        for app in class_apps:
            assert app.demand >= app_id_to_allocation[app.app_id]
        assert math.isclose(float((class_allocation)), 0.0, abs_tol=1e-3), class_allocation

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

                if math.isclose(float(allocation), 0, abs_tol=1e-3):
                    allocation = 0

                class_allocation[i] += allocation
                class_demand[i] -= allocation
                residual -= allocation
                total_demand -= allocation

                if math.isclose(float(class_demand[i]), 0, abs_tol=1e-3):
                    self._class_rates[i] = 0
                    class_demand[i] = 0

            if math.isclose(float(residual), 0.0, abs_tol=1e-3) or math.isclose(
                float(total_demand), 0.0, abs_tol=1e-3):
                break

            R = sum(self._class_rates)
            if R > 0:
                self._class_rates = [
                    frac(residual * self._class_rates[i], R)
                    for i in range(self._num_classes)
                ]

            tries += 1

            if tries > 10:
                break
                # raise Exception("Too many while loops")

        # after this while loop, we have gpu allocations per class in the class_allocation vector

        self._class_rates = class_allocation[:]
        app_id_to_allocation = {}

        for app_class in range(self._num_classes):
            self.__intra_class_allocation(
                app_class, class_allocation[app_class], app_id_to_allocation
            )

        return app_id_to_allocation

    def compute_allocation(self, event_time):
        app_id_to_mcs_allocation = self.compute_MCS_allocation(event_time)

        residual = self._fractional_share
        app_id_to_allocation = {}

        # assign int allocation
        for app_id in self._app_id_to_int_allocation:
            app_id_to_allocation[app_id] = int(self._app_id_to_int_allocation[app_id])

        # these apps are those with fractional share
        total_remaining_demand = sum(
            [
                app.demand - app_id_to_allocation[app.app_id]
                for app in self._sharing_group
            ]
        )

        for app in self._sharing_group:
            remaining_demand = app.demand - app_id_to_allocation[app.app_id]
            additional_allocation = min(residual, remaining_demand, 1)

            app_id_to_allocation[app.app_id] += additional_allocation
            residual -= additional_allocation
            total_remaining_demand -= additional_allocation

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

        self._redivision_event = None
        # left shift
        if len(self._sharing_group) > 1:
            self._sharing_group.append(self._sharing_group.pop(0))

            if not math.isclose(float(total_allocation), 0.0, abs_tol=1e-3):
                next_app = self._sharing_group[0]

                next_redivision = self._quantum * float(
                    self._app_id_to_fractional_allocation[next_app.app_id]
                )

                assert float(next_redivision) >= 0

                self._redivision_event = Event(
                    event_id=0,
                    event_time=event.event_time
                    + timedelta(seconds=float(next_redivision)),
                    event_type="REDIVISION",
                )


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



        if self._estimator:
            p_of_estimate = min(5000.0/len(self._app_list), 1.0)

            if not ray.is_initialized():
                if ray.__version__ == '2.0.0.dev0':
                    ray.init(ignore_reinit_error=True, address="auto")
                elif ray.__version__ == '2.10.0':
                    ray.init(ignore_reinit_error=True, address="auto", runtime_env={"env_vars": {"PYTHONPATH": "${PYTHONPATH}:"+f"{os.path.dirname(__file__)}/"}})
                else:
                    print("Warning: Incompatible Ray version --- may result in erroneous behaviour")

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
                if event.event_type in [Event.APP_SUB, Event.JOB_END]:
                    self._app_id_to_MCS_allocation = {}
                    self._app_id_to_fractional_allocation = {}
                    self._app_id_to_int_allocation = {}
                    self._fractional_share = None
                    self._sharing_group = list()

                    self._app_id_to_MCS_allocation = self.compute_MCS_allocation(
                        event.event_time
                    )

                    for app_id in self._app_id_to_MCS_allocation:
                        self._app_id_to_fractional_allocation[
                            app_id
                        ] = self._app_id_to_MCS_allocation[app_id] - int(
                            self._app_id_to_MCS_allocation[app_id]
                        )

                        self._app_id_to_int_allocation[app_id] = int(
                            self._app_id_to_MCS_allocation[app_id]
                        )

                        if self._app_id_to_fractional_allocation[app_id] > 0:
                            self._sharing_group.append(self._app_list[app_id])

                    self._fractional_share = int(
                        sum(
                            [
                                self._app_id_to_fractional_allocation[app_id]
                                for app_id in self._app_id_to_fractional_allocation
                            ]
                        )
                    )

                self.redivision(event)

            self.update_allocations(event.event_time)

            self.update_end_events(event.event_time)



            if event.event_type == Event.APP_SUB and self._estimator and random.uniform(0,1) < p_of_estimate:


                ret = self.sim_estimate(app=self._app_list[event.app_id], event_time=event.event_time)
                if ret:
                    self._sim_futures.append(ret)

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