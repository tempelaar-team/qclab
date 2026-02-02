"""
This module contains the parallel driver using the multiprocessing library.
"""

import multiprocessing
import logging
import copy
import numpy as np
import qclab.dynamics as dynamics
from qclab.utils import get_log_output, reset_log_output
from qclab import Data

logger = logging.getLogger(__name__)


def parallel_driver_multiprocessing(sim, seeds=None, data=None, num_tasks=None):
    """
    Parallel driver for the dynamics core using the python library multiprocessing.

    .. rubric:: Args
    sim: Simulation
        The Simulation object containing the model, algorithm, initial state, and settings.
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
    local_input_data = [
        (
            copy.deepcopy(sim),
            {"seed": batch_seeds_list[n][~np.isnan(batch_seeds_list[n])].astype(int)},
            {},
            Data(batch_seeds_list[n][~np.isnan(batch_seeds_list[n])].astype(int)),
        )
        for n in range(num_batches)
    ]
    for i in range(num_batches):
        # Determine the batch size from the seeds in the State object.
        local_input_data[i][0].settings.batch_size = len(local_input_data[i][1]["seed"])
        logger.info(
            "Running batch %s with seeds %s.", i + 1, local_input_data[i][1]["seed"]
        )
    logger.info("Starting dynamics calculation.")
    with multiprocessing.Pool(processes=size) as pool:
        results = pool.starmap(dynamics.run_dynamics, local_input_data)
    logger.info("Dynamics calculation completed.")
    logger.info("Collecting results from all tasks.")
    for result in results:
        data.add_data(result)
    logger.info("Simulation complete.")
    # Attach collected log output.
    data.log = get_log_output()
    return data
