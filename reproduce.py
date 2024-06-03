"""
New script to reproduce the results. 

Usage:
    Run reproduce.py -h using your Python interpreter (python or python3).

Environment:
    Please follow the instructions in the README.
"""

import argparse
import sys
import os

SIMULATION = "simulation"
UTIL = "utils"

# This is where majority of source code is
sys.path.insert(0, SIMULATION)
sys.path.insert(1, os.path.join(SIMULATION, UTIL))

# Get the necessary modules from primary source code
import result_summary
import preference_selector
import sim

# Unpickled objects are of these types
from wfq_tuner import Objective, FlexTuneWHeuristics

# Pareto configuration files, trace files, workload files, are stored here
PARETO_FRONTS_DIR = os.path.join(SIMULATION, "wfq_configurations")
TRACES_DIR = os.path.join(SIMULATION, "traces")
WORKLOADS_DIR = os.path.join(SIMULATION, "workloads")

# MCS config produced by preference_selector is stored here
MCS_CONFIG_FILE = os.path.join(SIMULATION, ".MCS_config.pkl")


def get_pareto_front_names():
    """
    Gets the names of pareto fronts (as specified in PARETO_FRONTS_DIR)

    e.g. "WFQ_trace_0e4a51_avg_pred_error_vs_avg_jct.pkl" -> "trace_0e4a51"
    """

    return {e[len("WFQ_") : e.index("_avg")] for e in os.listdir(PARETO_FRONTS_DIR)}


def get_trace_names():
    """
    Gets the trace names as specified in traces folder

    e.g. "trace_0e4a51.csv" -> "trace_0e4a51"
    """

    return {e[: -len(".csv")] for e in os.listdir(TRACES_DIR)}


def get_workload_names():
    """
    Gets the workload names as specified in workloads folder

    e.g. "cdf-app-interarrival-times-workload_0e4a51.csv" -> "workload_0e4a51"
    """

    return {e[e.rindex("-") + 1 : -len(".csv")] for e in os.listdir(WORKLOADS_DIR)}


def get_args():
    """Gets the pareto front name of user's choice"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pareto_front_filename",
        help="Pareto front file name",
        choices=get_pareto_front_names(),
    )
    parser.add_argument(
        "-models",
        choices=["realistic", "toy", "linear"],
        default="realistic",
        help="which model type to use",
    )
    parser.add_argument("-load", help="load", type=float, default=0.8)
    parser.add_argument("-num_gpus", help="num_gpus", default=64, type=int)
    parser.add_argument(
        "-num_apps", help="number of apps to generate", type=int, default=10
    )
    parser.add_argument(
        "-scheduling_policy", help="Scheduling policy", type=str, default="MCS"
    )
    parser.add_argument("-logging", help="logging verbosity (0-2)", default=1, type=int)
    parser.add_argument(
        "-estimate",
        help="whether to estimate ACTs 0/1",
        default=1,
        type=int,
        choices=[0, 1],
    )
    parser.add_argument("-output_file", default=None, type=str)
    parser.add_argument("-seed", type=int, default=4567)
    parser.add_argument("-p_error", type=float, default=None)
    args = parser.parse_args()
    if args.output_file is None:
        args.output_file = f"MCS_result_{args.pareto_front_filename}.csv"
    return args


def get_pareto_front_pickle_file(args):
    """Converts a pareto front filename -> pickle file name"""

    return os.path.join(
        PARETO_FRONTS_DIR,
        f"WFQ_{args.pareto_front_filename}_avg_pred_error_vs_avg_jct.pkl",
    )


def cleanup(args):
    """Deletes file produced by preference_solver, sim if it was made"""

    for f in [args.output_file, os.path.basename(MCS_CONFIG_FILE)]:
        if os.path.isfile(f):
            os.remove(f)


def main():
    args = get_args()

    if args.pareto_front_filename in get_trace_names():
        args.trace = args.pareto_front_filename
    elif args.pareto_front_filename in get_workload_names():
        args.workload = args.pareto_front_filename
    else:
        raise RuntimeError(
            f"{args.pareto_front_filename} is neither a trace nor workload"
        )
    args.MCS_config_file = os.path.basename(MCS_CONFIG_FILE)

    preference_selector.main(
        argparse.Namespace(
            fname=get_pareto_front_pickle_file(args),
            interactive=1,
            output_fname=MCS_CONFIG_FILE,
        )
    )

    if os.path.isfile(MCS_CONFIG_FILE):
        os.chdir(SIMULATION)
        sim.run_sim(args)
        result_summary.main(argparse.Namespace(fnames=[args.output_file]))
        cleanup(args)


if __name__ == "__main__":
    main()
