import logging
from typing import Dict, Any, Optional, Set, Union, Callable

import pickle
import warnings
import math

from ray.tune import trial_runner
from ray.tune.resources import Resources
from ray.tune.schedulers.resource_changing_scheduler import ResourceChangingScheduler
from ray.tune.trial import Trial
from ray.tune.utils.placement_groups import PlacementGroupFactory

logger = logging.getLogger(__name__)


class MyResourceChangingScheduler(ResourceChangingScheduler):
   
    def __init__(
            self,
            base_scheduler: Optional[TrialScheduler] = None,
            resources_allocation_function: Optional[Callable[[
                "trial_runner.TrialRunner", Trial, Dict[str, Any],
                "ResourceChangingScheduler"
            ], Union[None, PlacementGroupFactory,
                     Resources]]] = evenly_distribute_cpus_gpus,
    ) -> None:
        super().__init__()
        if resources_allocation_function is None:
            warnings.warn(
                "`resources_allocation_function` is None. No resource "
                "requirements will be changed at any time. Pass a "
                "correctly defined function to enable functionality.")
        self._resources_allocation_function = resources_allocation_function
        self._base_scheduler = base_scheduler or FIFOScheduler()
        self._base_trial_resources: Optional[Union[
            Resources, PlacementGroupFactory]] = None
        self._trials_to_reallocate: Dict[Trial, Union[
            None, dict, PlacementGroupFactory]] = {}
        self._reallocated_trial_ids: Set[str] = set()

    @property
    def metric(self):
        return self._base_scheduler._metric

    @property
    def base_trial_resources(
            self) -> Optional[Union[Resources, PlacementGroupFactory]]:
        return self._base_trial_resources

    def set_search_properties(self, metric: Optional[str],
                              mode: Optional[str]) -> bool:
        return self._base_scheduler.set_search_properties(metric, mode)

    def on_trial_add(self, trial_runner: "trial_runner.TrialRunner",
                     trial: Trial, **kwargs):
        # use the first trial resources as the base
        if self._base_trial_resources is None:
            if trial.uses_placement_groups:
                self._base_trial_resources = trial.placement_group_factory
            else:
                self._base_trial_resources = trial.resources
        # Raise error if the resources of a newly added trial don't match
        # base resources, but allow trials that have already had their
        # resources changed by ResourceChangingScheduler
        # (those can be added again during loading from a checkpoint)
        elif trial.trial_id not in self._reallocated_trial_ids:
            if trial.uses_placement_groups:
                trial_resources = trial.placement_group_factory
            else:
                trial_resources = trial.resources
            if trial_resources != self._base_trial_resources:
                raise RuntimeError(
                    "ResourceChangingScheduler doesn't support trials with "
                    "varying base resources. First trial had "
                    f"{self._base_trial_resources}, trial {trial} has "
                    f"{trial_resources}.")

        return self._base_scheduler.on_trial_add(trial_runner, trial, **kwargs)

    def on_trial_error(self, trial_runner: "trial_runner.TrialRunner",
                       trial: Trial, **kwargs):
        return self._base_scheduler.on_trial_error(trial_runner, trial,
                                                   **kwargs)

    def on_trial_result(self, trial_runner: "trial_runner.TrialRunner",
                        trial: Trial, result: Dict) -> str:
        base_scheduler_decision = self._base_scheduler.on_trial_result(
            trial_runner, trial, result)
        if base_scheduler_decision == TrialScheduler.CONTINUE:
            new_resources = self.reallocate_trial_resources_if_needed(
                trial_runner, trial, result)
            if new_resources:
                self._trials_to_reallocate[trial] = new_resources
                return TrialScheduler.PAUSE
        return base_scheduler_decision

    def on_trial_complete(self, trial_runner: "trial_runner.TrialRunner",
                          trial: Trial, result: Dict, **kwargs):
        return self._base_scheduler.on_trial_complete(trial_runner, trial,
                                                      result, **kwargs)

    def on_trial_remove(self, trial_runner: "trial_runner.TrialRunner",
                        trial: Trial, **kwargs):
        return self._base_scheduler.on_trial_remove(trial_runner, trial,
                                                    **kwargs)

    def choose_trial_to_run(self, trial_runner: "trial_runner.TrialRunner",
                            **kwargs) -> Optional[Trial]:
        if getattr(trial_runner.trial_executor, "_reuse_actors", False):
            raise ValueError("ResourceChangingScheduler cannot be used with "
                             "`reuse_actors=True`. FIX THIS by setting "
                             "`reuse_actors=False` in `tune.run`.")

        any_resources_changed = False

        new_trials_to_reallocate = {}
        for trial, new_resources in self._trials_to_reallocate.items():
            if trial.status == Trial.RUNNING:
                new_trials_to_reallocate[trial] = new_resources
                logger.debug(f"{trial} is still running, skipping for now")
                continue
            any_resources_changed = (any_resources_changed
                                     or self.set_trial_resources(
                                         trial, new_resources))
        self._trials_to_reallocate = new_trials_to_reallocate

        if any_resources_changed:
            # force reconcilation to ensure resource changes
            # are implemented right away
            trial_runner.trial_executor.force_reconcilation_on_next_step_end()

        trial = self._base_scheduler.choose_trial_to_run(
            trial_runner, **kwargs)
        return trial

    def debug_string(self) -> str:
        return ("(ResourceChangingScheduler) "
                f"{self._base_scheduler.debug_string()}")

    def save(self, checkpoint_path: str):
        save_object = self.__dict__
        with open(checkpoint_path, "wb") as outputFile:
            pickle.dump(save_object, outputFile)

    def restore(self, checkpoint_path: str):
        with open(checkpoint_path, "rb") as inputFile:
            save_object = pickle.load(inputFile)
        self.__dict__.update(save_object)

    def set_trial_resources(
            self, trial: Trial,
            new_resources: Union[Dict, Callable, PlacementGroupFactory]
    ) -> bool:
        """Returns True if new_resources were set."""
        if new_resources:
            logger.info(f"Setting trial {trial} resource to {new_resources}")
            trial.placement_group_factory = None
            trial.update_resources(new_resources)
            # keep track of all trials which had their resources changed
            self._reallocated_trial_ids.add(trial.trial_id)
            return True
        return False

    def _are_resources_the_same(
            self,
            trial: Trial,
            new_resources,
    ) -> bool:
        """Returns True if trial's resources are value equal to new_resources.

        Only checks for PlacementGroupFactories at this moment.
        """
        if trial.uses_placement_groups:
            if (isinstance(new_resources, PlacementGroupFactory)
                    and trial.placement_group_factory == new_resources):
                logger.debug(
                    f"{trial} PGF "
                    f"{trial.placement_group_factory.required_resources}"
                    f" and {new_resources.required_resources}"
                    f" are the same, skipping")
                return True
        return False

    def reallocate_trial_resources_if_needed(
            self, trial_runner: "trial_runner.TrialRunner", trial: Trial,
            result: Dict) -> Union[None, dict, PlacementGroupFactory]:
        """Calls user defined resources_allocation_function. If the returned
        resources are not none and not the same as currently present, returns
        them. Otherwise, returns None."""
        if self._resources_allocation_function is None:
            return None

        new_resources = self._resources_allocation_function(
            trial_runner, trial, result, self)

        # if we can check if the new resources are the same,
        # we do that here and skip resource allocation
        if new_resources and not self._are_resources_the_same(
                trial, new_resources):
            return new_resources
        return None

