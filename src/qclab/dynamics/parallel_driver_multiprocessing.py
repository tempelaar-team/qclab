"""
This module contains the parallel driver using the multiprocessing library.
"""

import multiprocessing
import logging
import copy
import time
import threading
import ctypes
import numpy as np
import qclab.dynamics as dynamics
from qclab.utils import get_log_output, reset_log_output
from qclab import Data
from qclab.dynamics.progress import (
    BatchProgressBars,
    SharedValueProgressReporter,
)

logger = logging.getLogger(__name__)


def parallel_driver_multiprocessing(sim, seeds=None, data=None, num_tasks=None):
    """
    Parallel driver for the dynamics core using the python library multiprocessing.

    .. rubric:: Args
    sim: Simulation
        The simulation object containing the model, algorithm, initial state, and settings.
    seeds: ndarray, optional
        An array of integer seeds for the trajectories. If None, seeds will be
        generated automatically.
    data: Data, optional
        A Data object for collecting output data. If None, a new Data object
        will be created.
    num_tasks: int, optional
        The number of tasks to use for parallel processing. If None, the
        number of available tasks will be used.

    .. rubric:: Returns
    data: Data
        The updated Data object containing collected output data.
    """
    # Clear any in-memory log output from previous runs.
    reset_log_output()
    # First initialize the model constants.
    sim.model.initialize_constants()
    if data is None:
        data = Data()
    if seeds is None:
        if len(data.data_dict["seed"]) > 0:
            offset = np.max(data.data_dict["seed"]) + 1
        else:
            offset = 0
        seeds = offset + np.arange(sim.settings.num_trajs, dtype=int)
        num_trajs = sim.settings.num_trajs
    else:
        num_trajs = len(seeds)
        logger.warning(
            "Setting sim.settings.num_trajs to the number of provided seeds: %s",
            num_trajs,
        )
        sim.settings.num_trajs = num_trajs
    if num_tasks is None:
        size = multiprocessing.cpu_count()
    else:
        size = num_tasks
    logger.info("Using %s tasks for parallel processing.", size)
    # Determine the number of batches required to execute the total number
    # of trajectories.
    if num_trajs % sim.settings.batch_size == 0:
        num_batches = num_trajs // sim.settings.batch_size
    else:
        num_batches = num_trajs // sim.settings.batch_size + 1
    logger.info(
        "Running %s batches with %s seeds in each batch.",
        num_batches,
        sim.settings.batch_size,
    )
    batch_seeds_list = (
        np.zeros((num_batches * sim.settings.batch_size), dtype=int) + np.nan
    )
    batch_seeds_list[:num_trajs] = seeds
    batch_seeds_list = batch_seeds_list.reshape((num_batches, sim.settings.batch_size))
    # Create the input data for each local simulation.
    sim.initialize_timesteps()
    steps_per_batch = len(sim.settings.t_update_n)
    progress_enabled = getattr(sim.settings, "progress_bar", True)
    batch_bars = BatchProgressBars(
        num_batches, steps_per_batch, enabled=progress_enabled
    )
    progress_values = []
    local_input_data = [
        (
            copy.deepcopy(sim),
            {"seed": batch_seeds_list[n][~np.isnan(batch_seeds_list[n])].astype(int)},
            {},
            Data(batch_seeds_list[n][~np.isnan(batch_seeds_list[n])].astype(int)),
        )
        for n in range(num_batches)
    ]
    if progress_enabled:
        progress_values = [
            multiprocessing.Value(ctypes.c_int, 0) for _ in range(num_batches)
        ]
    for i in range(num_batches):
        # Determine the batch size from the seeds in the state object.
        local_input_data[i][0].settings.batch_size = len(local_input_data[i][1]["seed"])
        logger.info(
            "Running batch %s with seeds %s.", i + 1, local_input_data[i][1]["seed"]
        )
    logger.info("Starting dynamics calculation.")
    monitor_thread = None
    stop_event = threading.Event()
    if progress_enabled and progress_values:
        def _monitor_progress():
            completed = 0
            while not stop_event.is_set() and completed < num_batches:
                snapshot = [val.value for val in progress_values]
                completed = sum(1 for val in snapshot if val >= steps_per_batch)
                running = [val for val in snapshot if val < steps_per_batch]
                slowest = min(running) if running else steps_per_batch
                batch_bars.set_total(completed)
                batch_bars.set_slowest(slowest)
                time.sleep(0.1)
            batch_bars.set_total(num_batches)
            batch_bars.set_slowest(steps_per_batch)

        monitor_thread = threading.Thread(target=_monitor_progress, daemon=True)
        monitor_thread.start()
    worker_inputs = [
        local_input_data[i]
        + (
            SharedValueProgressReporter(progress_values[i], steps_per_batch)
            if progress_enabled
            else None,
        )
        for i in range(num_batches)
    ]
    with multiprocessing.Pool(processes=size) as pool:
        results = pool.starmap(_run_batch_with_progress, worker_inputs)
    stop_event.set()
    if monitor_thread is not None:
        monitor_thread.join()
    logger.info("Dynamics calculation completed.")
    logger.info("Collecting results from all tasks.")
    for result in results:
        data.add_data(result)
    if progress_enabled:
        batch_bars.close()
    logger.info("Simulation complete.")
    # Attach collected log output.
    data.log = get_log_output()
    return data


def _run_batch_with_progress(sim, state, parameters, data, progress_reporter=None):
    """Execute a single batch with an optional progress reporter."""

    return dynamics.run_dynamics(
        sim, state, parameters, data, progress_reporter=progress_reporter
    )
