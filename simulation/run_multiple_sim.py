import os
import sys
import argparse
import ray
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Union
from enum import Enum
import uuid

import pickle
import datetime



CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..' , 'data', 'PCS_configs')) 
RESULT_SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..' , 'new_data')) 


from sim import (
    run_sim,
    TRACES,
    WORKLOADS,
)


@dataclass(eq=True, frozen=True)
class SimArgs:
    num_apps:int
    num_gpus: int
    workload: str
    load: float
    seed: int
    output_file: str
    scheduling_policy: str
    MCS_config_file: Union[str, None]
    p_error: Union[float, None] = None
    estimate: int = 1
    models: str = "linear"
    logging: int = 1
    trace: Union[str, None] = None


@ray.remote
def remote_runner(func, args):
    tick = datetime.datetime.now()
    res = func(args)
    tock = datetime.datetime.now()
    return {"result": res, "total_time": (tock-tick).total_seconds()}


def run_multiple(args):
    if not ray.is_initialized():
        if ray.__version__ == '2.0.0.dev0':
            ray.init(ignore_reinit_error=True, address="auto")
        elif ray.__version__ == '2.10.0':
            ray.init(ignore_reinit_error=True, address="auto", runtime_env={"env_vars": {"PYTHONPATH": "${PYTHONPATH}:"+f"{os.path.dirname(__file__)}/"}})
        else:
            print("Warning: Untested Ray version --- may result in erroneous behaviour")

    futures = {}

    for n in args.num_apps:
        for g in args.num_gpus:
            for l in args.loads:
                for s in args.seeds:
                    for w in args.workloads:
                        for p in args.scheduling_policies:                            
                            for c in args.PCS_configs if p == 'MCS' else [None]:
                                for e in args.p_error:
                                    sim_args = SimArgs(n,g,w,l,s, os.path.join(RESULT_SAVE_PATH,f"{str(uuid.uuid4())}.csv"),p,c,e)
                                    futures[sim_args] = remote_runner.remote(run_sim, sim_args)

    results = {}

    for k,v in futures.items():
        results[k] = ray.get(v)

    # SYNCING RESULT FILES - This will overwrite stuff:

    try:
        workers = ray.nodes()
        for w in workers:
            w_ip = w.get('NodeManagerAddress')
            os.system(f"rsync -av {w_ip}:{RESULT_SAVE_PATH}/*.csv {RESULT_SAVE_PATH}/")
    except Exception as e:
        print(f"Error: unable to sync files across worker nodes: {e} --- please do it manually")

    return results

if __name__ == '__main__':
    
    # when MCS specified in policies, it is the callers responsibility to ensure valid configs are passed
    # all configs will be used for all policies

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-output_file", help=".pkl output file to store summary", default = "output.pkl", type=str
    )

    
    parser.add_argument(
        "-num_apps", nargs="+", help="space seperated num_apps to try.", default = [512], type=int
    )

    parser.add_argument(
        "-num_gpus", nargs="+", help="space seperated num_gpus to try.", default = [64], type=int
    )

    parser.add_argument(
        "-loads", nargs="+", help="space seperated loads to try.", default = [0.8], type=float
    )

    parser.add_argument(
        "-seeds", nargs="+", help="space seperated seeds to try.", default = [4567], type=int
    )

    parser.add_argument(
        "-workloads", nargs="+", help="space seperated workloads to try.", default = ["themis1"], type=str
    )

    parser.add_argument(
        "-scheduling_policies", nargs="+", help="space seperated scheduling policies to try.", default = ["MCS"], type=str
    )

    parser.add_argument(
        "-PCS_configs", nargs="+", help="space seperated MCS_configs to try.", default = [os.path.join(CONFIG_PATH, 'PCS_config_themis1_bal.pkl')], type=str
    )

    parser.add_argument(
        "-p_error", nargs="+", help="space seperated error percentage to try.", default = [0.0], type=float
    )


    args = parser.parse_args()
    results = run_multiple(args)

    with open(args.output_file, 'wb') as fp:
        pickle.dump(results, fp)