import logging
from typing import Dict, Optional, Union
import sys

import numpy as np
import pickle

from ray.tune import trial_runner
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.trial import Trial

from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class TimedFIFOScheduler(FIFOScheduler):
    """Simple scheduler that just runs trials in submission order."""

    def __init__(self,
                time_attr: str = "training_iteration",
                budget: float = 100.0):

        self._budget = budget
        self._time_attr = time_attr

        self._trial_id_to_time_elapsed = {}
        self._trial_id_to_last_event_time = {}
        self._trial_id_to_budget = {}


    def on_trial_pause(self, trial_runner: "trial_runner.TrialRunner",
                          trial: Trial):

        self._trial_id_to_last_event_time[trial.trial_id] = None



    def on_trial_unpause(self, trial_runner: "trial_runner.TrialRunner",
                          trial: Trial):

        self._trial_id_to_last_event_time[trial.trial_id] = datetime.now()


    def on_trial_add(self, trial_runner: "trial_runner.TrialRunner",
                     trial: Trial):
        pass


    def on_trial_start(self, trial_runner: "trial_runner.TrialRunner",
                     trial: Trial):
        self._trial_id_to_time_elapsed[trial.trial_id] = 0
        self._trial_id_to_last_event_time[trial.trial_id] = datetime.now()
        self._trial_id_to_budget[trial.trial_id] = self._budget




    def on_trial_result(self, trial_runner: "trial_runner.TrialRunner",
                        trial: Trial, result: Dict) -> str:

        
        if self._trial_id_to_last_event_time[trial.trial_id] != None:
            now_time = datetime.now()
            self._trial_id_to_time_elapsed[trial.trial_id] += (now_time - self._trial_id_to_last_event_time[trial.trial_id]).total_seconds()
            self._trial_id_to_last_event_time[trial.trial_id] = now_time

        if self._trial_id_to_time_elapsed[trial.trial_id] <= self._budget:
            action = TrialScheduler.CONTINUE
        else:
            action = TrialScheduler.STOP

        return action



    def choose_trial_to_run(
            self, trial_runner: "trial_runner.TrialRunner") -> Optional[Trial]:
        for trial in trial_runner.get_trials():
            if (trial.status == Trial.PENDING):
                return trial
        for trial in trial_runner.get_trials():
            if (trial.status == Trial.PAUSED):
                return trial
        return None


    def debug_string(self) -> str:
        return "Using TimedFIFO scheduling algorithm."


    def estimate_remaining_trial_times(self):
        trial_id_to_estimated_remaining_time = {}
        for trial_id in self._trial_id_to_time_elapsed:
            trial_id_to_estimated_remaining_time[trial_id] = max(0, self._trial_id_to_budget[trial_id] - self._trial_id_to_time_elapsed[trial_id])
        return trial_id_to_estimated_remaining_time

        

