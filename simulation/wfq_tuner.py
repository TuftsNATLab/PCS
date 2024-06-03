from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.termination_criterion import StoppingByTime

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.evaluator import MultiprocessEvaluator, RayEvaluator


import numpy as np
import os, sys, argparse
import csv
import copy, bisect
import matplotlib.pyplot as plt
import pickle
from fractions import Fraction as frac
from datetime import datetime, timedelta
from functools import partial


from models import Models
from MCSScheduler import AppMCScheduler
from helpers import gen_data_from_cdf
from common import App, Job, Event
from sim import *


import pickle
import argparse
import ray

# In[3]:


class Objective(object):
    """docstring for Objective"""

    def __init__(self, lmda, label):
        super(Objective, self).__init__()
        self._lmda = lmda
        self._label = label

    def __call__(self, entry):
        return self._lmda(entry)

    def get_name(self):
        return self._label


class FlexTune(FloatProblem):
    """docstring for FlexTune"""

    def __init__(self, total_gpus, app_list, event_queue, objectives):
        super(FlexTune, self).__init__()

        self._total_gpus = total_gpus
        self._app_list = app_list
        self._event_queue = event_queue

        self._service_times = list()
        for app_id in app_list:
            self._service_times.append(app_list[app_id].service)
        self._service_times.sort()

        self.objectives = objectives[:]
        self.number_of_objectives = len(objectives)
        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = [objective.get_name() for objective in objectives]

        # ['mean_pred','mean_jct']

    def get_bounds(self):
        return [self.lower_bound, self.upper_bound]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        try:
            class_detail = self.solution_transformer(solution)

        except:
            # print(f"solution {solution} cannot be transformed")

            solution.objectives = [float("inf")] * self.number_of_objectives

            return solution

        # print(f"solution: {solution} class_detail: {class_detail}")

        scheduler = AppMCScheduler(
            total_gpus=self._total_gpus,
            event_queue=copy.deepcopy(self._event_queue),
            app_list=copy.deepcopy(self._app_list),
            class_detail=class_detail,
            app_info_fn=None,
            verbosity=0,
        )
        scheduler.set_estimator()

        tick = datetime.now()
        scheduler.run()
        tock = datetime.now()

        objectives = self.__get_objective_value(scheduler, True)

        solution.objectives = objectives[:]

        return solution

    def __get_objective_value(self, scheduler, estimate):
        jct = list()
        pred_error = list()
        unfairness = list()

        app_list = scheduler._app_list

        for app_id in app_list:
            app = app_list[app_id]

            actual_jct = (app.end_time - app.submit_time).total_seconds()
            jct.append(actual_jct)

            if estimate and len(app.estimated_end_time) > 0:
                estimated_jct = (
                    app.estimated_end_time[0] - app.submit_time
                ).total_seconds()
                pred_error.append(
                    100.0 * abs(estimated_jct - actual_jct) / estimated_jct
                )

            num_apps_seen_diff = app.num_apps_seen[0] / app.num_apps_seen[1]
            divided_cluster_size = scheduler._max_capacity / num_apps_seen_diff
            fair_jct = app.service / min(divided_cluster_size, app.initial_demand)
            unfairness.append(max(0, ((actual_jct / fair_jct) - 1.0)))

        # 0, 1, 2
        jct.sort()
        pred_error.sort()
        unfairness.sort()

        # jct, unfairness, pred error

        obj_vals = list()

        for objective in self.objectives:
            if "pred" in objective.get_name():
                # print(f"{objective.get_name()}: {objective(pred_error)}")
                obj_vals.append(objective(pred_error))

            elif "jct" in objective.get_name():
                obj_vals.append(objective(jct))
                # print(f"{objective.get_name()}: {objective(jct)}")

            elif "unfairness" in objective.get_name():
                obj_vals.append(objective(unfairness))
                # print(f"{objective.get_name()}: {objective(unfairness)}")

        return obj_vals

    def get_name(self) -> str:
        raise NotImplementedError

    def solution_transformer(self, solution):
        raise NotImplementedError


class FlexTuneWoHeuristics(FlexTune):
    """docstring for FlexTuneWoHeuristics"""

    def __init__(self, total_gpus, app_list, event_queue, objectives):
        super(FlexTuneWoHeuristics, self).__init__(
            total_gpus, app_list, event_queue, objectives
        )

        self.number_of_variables = 13
        self.number_of_constraints = 0

        # num_classes
        # T1,T2,T3,T4,T5
        # R1,R2,R3,R4,R5
        # Clip_factor
        self.lower_bound = [1.0] + [1.0] * 5 + [0.0] * 5 + [0.0] + [0.01]
        self.upper_bound = (
            [5.5] + [max(self._service_times)] * 5 + [1.0] * 5 + [1.0] + [0.9]
        )

    def get_name(self) -> str:
        return "FlexTuneWoHeuristics"

    def solution_transformer(self, solution):
        num_classes = int(solution.variables[0])
        Ts = solution.variables[1 : 1 + num_classes]
        Rs = solution.variables[6 : 6 + num_classes]
        clip_demand_factor = solution.variables[-2]
        delta = solution.variables[-1]

        thresholds = self.__eval_T(Ts)
        rates = self.__eval_R(Rs)

        class_detail = {
            "num_classes": len(thresholds),
            "class_thresholds": thresholds,
            "class_rates": rates,
            "clip_demand_factor": clip_demand_factor,
            "delta": delta,
        }

        return class_detail

    def __eval_R(self, Rs):
        S = sum(Rs)
        rates = list(map(lambda r: r / sum(Rs), Rs))

        rates = list(map(lambda r: frac(round(r, 3)).limit_denominator(10000), rates))
        rates = list(map(lambda r: frac(r, sum(rates)), rates))

        rates[-1] = frac(1, 1) - sum(rates[:-1])

        assert all(list(map(lambda r: r >= 0 and r <= 1.0, rates)))
        assert np.isclose(float(sum(rates)), 1.0)
        return rates

    def __eval_T(self, Ts):
        thresholds = sorted(Ts)
        thresholds[-1] = float("inf")
        return thresholds


class WFQTuneWoHeuristics(FlexTuneWoHeuristics):
    ###################################################################
    # This can be thought of as a typedef of FlexTuneWoHeuristics
    # its used just so we can create pickle files of this type
    # from older files that used FlexTuneWoHeuristics while still 
    # maintaining the capability of reading old files that use 
    # FlexTuneWoHeuristics.
    # 11/29/23
    ###################################################################

    def __init__(self, total_gpus, app_list, event_queue, objectives):
        super().__init__(total_gpus, app_list, event_queue, objectives)


def comp_thresholds(job_sizes, cov_thresh=1.0):
    thresholds = []

    n = 0
    mu = 0
    s = 0
    s2 = 0
    cov = 0

    class_num = 0
    cov_history = list()

    for i in range(len(job_sizes)):
        xn = job_sizes[i]
        s += xn
        s2 += xn * xn
        n += 1
        mu = ((n - 1.0) * mu + xn) / n
        var = (1.0 / n) * (s2 + (n * mu * mu) - (mu * 2.0 * s))

        cov = var / (mu * mu)

        cov_history.append(cov)

        if cov > cov_thresh:
            thresholds.append(int(job_sizes[i - 1]))
            n = 0
            mu = 0
            s = 0
            s2 = 0
            class_num += 1

    thresholds = thresholds + [float("inf")]
    return thresholds, cov, cov_history


class FlexTuneWHeuristics(FlexTune):
    """docstring for FlexTuneWHeuristics"""

    def __init__(self, total_gpus, app_list, event_queue, objectives):
        super(FlexTuneWHeuristics, self).__init__(
            total_gpus, app_list, event_queue, objectives
        )
        self.number_of_variables = 4
        self.number_of_constraints = 0

        _, _, cov_history = comp_thresholds(
            self._service_times, cov_thresh=float("inf")
        )

        # T, R, Clip_factor
        self.lower_bound = [0.001, -3.0, 0.0, 0.01]
        self.upper_bound = [max(cov_history) + 1.0, 3.0, 1.0, 0.9]

    def get_name(self) -> str:
        return "FlexTuneWHeuristics"

    def solution_transformer(self, solution):
        T = solution.variables[0]
        R = solution.variables[1]
        clip_demand_factor = solution.variables[2]
        delta = solution.variables[3]

        thresholds = self.__eval_T(T)
        rates = self.__eval_R(R, len(thresholds))

        class_detail = {
            "num_classes": len(thresholds),
            "class_thresholds": thresholds,
            "class_rates": rates,
            "clip_demand_factor": clip_demand_factor,
            "delta": delta,
        }

        return class_detail

    def __eval_R(self, R, num_classes):
        rates = [np.exp(-1.0 * R * i) for i in range(num_classes)]
        rates = list(map(lambda r: r / sum(rates), rates))

        rates = list(map(lambda r: frac(round(r, 3)).limit_denominator(10000), rates))
        rates = list(map(lambda r: frac(r, sum(rates)), rates))

        rates[-1] = frac(1, 1) - sum(rates[:-1])

        assert all(list(map(lambda r: r > 0 and r <= 1.0, rates)))
        assert np.isclose(float(sum(rates)), 1.0)
        return rates

    def __eval_T(self, T):
        thresholds, *_ = comp_thresholds(self._service_times, cov_thresh=T)
        return thresholds


class WFQTuneWHeuristics(FlexTuneWHeuristics):
    ###################################################################
    # This can be thought of as a typedef of FlexTuneWHeuristics
    # its used just so we can create pickle files of this type
    # from older files that used FlexTuneWHeuristics while still 
    # maintaining the capability of reading old files that use 
    # FlexTuneWHeuristics.
    # 11/29/23
    ###################################################################
    def __init__(self, total_gpus, app_list, event_queue, objectives):
        super().__init__(total_gpus, app_list, event_queue, objectives)


def get_results(scheduler, objectives):
    jct = list()
    pred_error = list()
    unfairness = list()

    app_list = scheduler._app_list

    for app_id in app_list:
        app = app_list[app_id]

        if len(app.estimated_end_time):
            actual_jct = (app.end_time - app.submit_time).total_seconds()
            jct.append(actual_jct)
            estimated_jct = (
                app.estimated_end_time[0] - app.submit_time
            ).total_seconds()
            pred_error.append(100.0 * abs(estimated_jct - actual_jct) / estimated_jct)

            num_apps_seen_diff = app.num_apps_seen[0] / app.num_apps_seen[1]
            divided_cluster_size = scheduler._max_capacity / num_apps_seen_diff
            fair_jct = app.service / min(divided_cluster_size, app.initial_demand)
            unfairness.append(max(0, ((actual_jct / fair_jct) - 1.0)))

    # 0, 1, 2
    jct.sort()
    pred_error.sort()
    unfairness.sort()

    # jct, unfairness, pred error

    obj_vals = list()

    for objective in objectives:
        if "pred" in objective.get_name():
            # print(f"{objective.get_name()}: {objective(pred_error)}")
            obj_vals.append(objective(pred_error))

        elif "jct" in objective.get_name():
            obj_vals.append(objective(jct))
            # print(f"{objective.get_name()}: {objective(jct)}")

        elif "unfairness" in objective.get_name():
            obj_vals.append(objective(unfairness))
            # print(f"{objective.get_name()}: {objective(unfairness)}")

    return obj_vals


@ray.remote
def sim(total_gpus, event_queue, app_list, problem, solution_id, solution):
    solution = copy.deepcopy(solution)
    class_detail = copy.deepcopy(problem.solution_transformer(solution))
    objectives = copy.deepcopy(problem.objectives)

    scheduler = AppMCScheduler(
        total_gpus=total_gpus,
        event_queue=copy.deepcopy(event_queue),
        app_list=copy.deepcopy(app_list),
        class_detail=class_detail,
        app_info_fn=None,
        estimate=True,
        verbosity=0,
    )

    scheduler.run()

    objectives = get_results(scheduler, objectives)

    solution.objectives = objectives[:]

    return {"solution_id": solution_id, "solution": solution}


def run_simulations_parallel(
    workload, load, total_gpus, num_apps, solutions, problem, models
):
    app_lists = {}
    event_queues = {}

    for seed in [617, 4298, 9470, 4580, 3438]:
        app_list = {}
        event_queue = list()

        gen_workload(
            workload,
            workload,
            workload,
            workload,
            load,
            total_gpus,
            num_apps,
            seed,
            app_list,
            event_queue,
            models,
        )

        app_lists[seed] = copy.deepcopy(app_list)
        event_queues[seed] = copy.deepcopy(event_queue)


    if not ray.is_initialized():
        if ray.__version__ == '2.0.0.dev0':
            ray.init(ignore_reinit_error=True, address="auto")
        elif ray.__version__ == '2.10.0':
            ray.init(ignore_reinit_error=True, address="auto", runtime_env={"env_vars": {"PYTHONPATH": "${PYTHONPATH}:"+f"{os.path.dirname(__file__)}/"}})
        else:
            print("Warning: Incompatible Ray version --- may result in erroneous behaviour")

    futures = list()

    solutions_dic = {}

    for solution_id, solution in enumerate(solutions):
        class_detail = problem.solution_transformer(solution)
        for seed in [617, 4298, 9470, 4580, 3438]:
            app_list = app_lists[seed]
            event_queue = event_queues[seed]

            futures.append(
                sim.remote(
                    total_gpus, event_queue, app_list, problem, solution_id, solution
                )
            )

        solutions_dic[solution_id] = list()

    while futures:
        # Returns the first ObjectRef that is ready.
        finished, futures = ray.wait(futures, num_returns=1)
        result = ray.get(finished[0])

        solutions_dic[result["solution_id"]].append(result["solution"])

    avg_solutions = list()
    number_of_objectives = problem.number_of_objectives
    for solutions in solutions_dic.values():
        avg_solution = copy.deepcopy(solutions[0])

        avg_solution.objectives_std = [0] * number_of_objectives

        for obj in range(number_of_objectives):
            d = [solution.objectives[obj] for solution in solutions]
            avg_solution.objectives[obj] = np.mean(d)
            avg_solution.objectives_std[obj] = np.std(d)

        avg_solutions.append(avg_solution)

    return avg_solutions, solutions_dic


def parse_objectives(objectives):
    objectives_list = list()

    allowed_metrics = ["pred_error", "jct", "unfairness"]

    if len(objectives) < 1:
        print("At least one objective must be specified")
        raise NotImplementedError

    for objective in objectives:
        measure, *metric = objective.split("_")

        metric = "_".join(metric)

        if metric not in allowed_metrics:
            print(f"Currently only {allowed_metrics} metrics allowed")
            raise NotImplementedError

        if measure == "avg":
            objectives_list.append(Objective(np.mean, objective))
        elif measure[0] == "p":
            print(float(measure[1:]))
            objectives_list.append(
                Objective(partial(np.percentile, q=float(measure[1:])), objective)
            )
        else:
            print(f"Currently only avg or pXX measure allowed")
            raise NotImplementedError

    return objectives_list




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs SPEA2 to find Pareto-optimal Flex configs"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-trace", help="trace name", choices=TRACES, type=str)
    group.add_argument("-workload", help="workload name", choices=WORKLOADS, type=str)
    parser.add_argument(
        "-models",
        choices=["realistic", "toy", "linear"],
        default="linear",
        help="which model type to use",
    )

    parser.add_argument(
        "-use_heuristics", type=int, help="use heuristics 1/0", default=1
    )
    parser.add_argument("-max_eval", type=int, help="max_eval", default=1280)
    parser.add_argument("-total_gpus", type=int, help="total_gpus", default=64)
    parser.add_argument("-load", type=float, help="load", default=0.8)
    parser.add_argument("-num_apps", type=int, help="num_apps", default=2000)
    parser.add_argument(
        "-fname", type=str, help="name of output pkl file", default=None
    )
    parser.add_argument("-checkpoint", type=int, help="checkpoint (1/0)", default=0)

    parser.add_argument(
        "-objectives", nargs="+", help="list of objectives", type=str, required=True
    )
    parser.add_argument(
        "-population_size", help="size of population", type=int, default=70
    )

    args = parser.parse_args()

    use_heuristics = args.use_heuristics
    load = args.load
    total_gpus = args.total_gpus
    num_apps = args.num_apps
    max_eval = args.max_eval
    fname = args.fname
    checkpoint = args.checkpoint

    objectives = parse_objectives(args.objectives)

    models = Models(args.models)

    seed = 4567

    app_list = {}
    event_queue = list()

    if args.trace:
        gen_workload_from_trace(
            args.trace, app_list, event_queue, models, max_apps=args.num_apps
        )
    else:
        gen_workload(
            args.workload,
            args.workload,
            args.workload,
            args.workload,
            args.load,
            args.total_gpus,
            args.num_apps,
            seed,
            app_list,
            event_queue,
            models,
        )

    if use_heuristics:
        problem = FlexTuneWHeuristics(total_gpus, app_list, event_queue, objectives)
    else:
        problem = FlexTuneWoHeuristics(total_gpus, app_list, event_queue, objectives)

    fname = fname or f"learnt_configs_{problem.get_name()}.pkl"

    experiment_info = {
        "objectives": objectives,
        "time": datetime.now(),
        "load": load,
        "total_gpus": total_gpus,
        "workload": args.trace if args.trace else args.workload,
    }

    if fname in os.listdir():
        print(f"warning - overwriting {fname}")
        ans = input("Do you want to proceed: Y/N?")

        if ans == "N":
            sys.exit(1)

    with open(fname, "wb") as fp:
        pickle.dump(experiment_info, fp)

    algorithm = SPEA2(
        problem=problem,
        population_size=args.population_size,
        offspring_population_size=int(args.population_size * 1.5),
        mutation=PolynomialMutation(
            probability=1.0 / problem.number_of_variables, distribution_index=20
        ),
        checkpoints=[2**i for i in range(8, 1 + int(np.log(max_eval) / np.log(2)))]
        if checkpoint
        else [],
        fname=fname,
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_eval),
        population_evaluator=RayEvaluator(processes=40),
    )

    tick = datetime.now()
    print(f"Started at {tick}")
    algorithm.run()
    solutions = get_non_dominated_solutions(algorithm.get_result())

    avg_std = False

    if avg_std and args.workload:
        print("obtaining avg and std of pareto front points")

        avg_solutions, solutions_dic = run_simulations_parallel(
            args.workload, load, total_gpus, num_apps, solutions, problem, models
        )

        final_checkpoint = {
            "PROBLEM": problem,
            "EVALUATIONS": max_eval,
            "SOLUTIONS": avg_solutions,
            "SOLUTION_DIC": solutions_dic,
            "COMPUTING_TIME": (datetime.now() - tick).total_seconds(),
        }

        with open(fname, "ab") as fp:
            pickle.dump(final_checkpoint, fp)

    print("done")
    print(f"took {(datetime.now() - tick).total_seconds()} sec")
