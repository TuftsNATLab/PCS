"""
This is a helper script that will take all the pkl files in the wfq_configurations folder
and any that have a PROBLEM of type FlexTuneWHeuristics or FlexTuneWoHeuristics will 
be converted to WFQTuneWHeuristics or WFQTuneWoHeuristics. 
11/29/23
"""

from wfq_tuner import *

import pickle
import os


def load_obj_list(fname):
    """Loads list of objects pickled with dump( ) as in preference_selector"""

    objs = list()
    with open(fname, "rb") as fp:
        while True:
            try:
                objs.append(pickle.load(fp))
            except EOFError:
                break
    return objs


def convert(obj):
    """Converts PROBLEM to be WFQ named instead of Flex"""

    old = obj["PROBLEM"]
    if isinstance(old, FlexTuneWHeuristics):
        obj["PROBLEM"] = WFQTuneWHeuristics(
            old._total_gpus, old._app_list, old._event_queue, old.objectives
        )
    elif isinstance(old, FlexTuneWoHeuristics):
        obj["PROBLEM"] = WFQTuneWoHeuristics(
            old._total_gpus, old._app_list, old._event_queue, old.objectives
        )


def store_obj_list(objs, fname):
    """Stores objects in same manner as expected by preference_solver (dump 1 by 1)"""

    with open(fname, "wb+") as f:
        for obj in objs:
            pickle.dump(obj, f)


def main():
    for file in os.listdir("wfq_configurations"):
        fname = os.path.join("wfq_configurations", file)
        objs = load_obj_list(fname)

        # The place where PROBLEM is are in the objects following header
        for obj in objs[1:]:
            convert(obj)
        store_obj_list(objs, fname)


if __name__ == "__main__":
    main()
