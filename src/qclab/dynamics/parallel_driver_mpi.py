"""
This module contains the parallel MPI driver.
"""

import logging
import copy
import numpy as np
import qclab.dynamics as dynamics
from qclab.utils import get_log_output, reset_log_output
from qclab import Data
from qclab.dynamics.progress import BatchProgressBars, SerialProgressReporter

logger = logging.getLogger(__name__)


def parallel_driver_mpi(sim, seeds=None, data=None, num_tasks=None):
    """
    Parallel driver for the dynamics core using the mpi4py library.

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
    try:
        from mpi4py import MPI
    except ImportError:
        raise ImportError(
            "The package mpi4py is required for the parallel_driver_mpi driver."
        ) from None
    except Exception as e:
        raise RuntimeError(f"An error occurred when importing mpi4py: {e}") from None
    if seeds is None:
        offset = 0
        if data is not None:
            if len(data.data_dict["seed"]) > 0:
                offset = np.max(data.data_dict["seed"]) + 1
        else:
            data = Data()
        seeds = offset + np.arange(sim.settings.num_trajs, dtype=int)
        num_trajs = sim.settings.num_trajs
    else:
        num_trajs = len(seeds)
        logger.warning(
            "Setting sim.settings.num_trajs to the number of provided seeds: %s",
            num_trajs,
        )
        sim.settings.num_trajs = num_trajs
        if data is None:
            data = Data()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if num_tasks is None:
        size = comm.Get_size()
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
    # Split the batches into chunks for each MPI process.
    chunk_inds = np.linspace(0, num_batches, size + 1, dtype=int)
    start = chunk_inds[rank]
    end = chunk_inds[rank + 1]
    chunk_size = end - start
    # Create the input data for each local simulation.
    sim.initialize_timesteps()
    local_input_data = [
        (
            copy.deepcopy(sim),
            {"seed": batch_seeds_list[n][~np.isnan(batch_seeds_list[n])].astype(int)},
            {},
            Data(batch_seeds_list[n][~np.isnan(batch_seeds_list[n])].astype(int)),
        )
        for n in np.arange(num_batches)[start:end]
    ]
    # Set the batch size for each local batch.
    for i in range(chunk_size):
        local_input_data[i][0].settings.batch_size = len(local_input_data[i][1]["seed"])
        logger.info(
            "Running batch %s with seeds %s.",
            i + 1 + start,
            local_input_data[i][1]["seed"],
        )
    # Execute the local batches.
    logger.info("Starting dynamics calculation.")
    sim.initialize_timesteps()
    steps_per_batch = len(sim.settings.t_update_n)
    progress_enabled = getattr(sim.settings, "progress_bar", True)
    batch_bars = BatchProgressBars(
        chunk_size, steps_per_batch, enabled=progress_enabled
    )
    local_results = []
    completed_batches = 0
    for batch in local_input_data:
        progress_reporter = (
            SerialProgressReporter(batch_bars if progress_enabled else None, steps_per_batch)
            if progress_enabled
            else None
        )
        local_results.append(
            dynamics.run_dynamics(
                batch[0], batch[1], batch[2], batch[3], progress_reporter=progress_reporter
            )
        )
        completed_batches += 1
        if progress_enabled:
            batch_bars.set_total(completed_batches)
            if completed_batches != chunk_size:
                batch_bars.set_slowest(0)
    comm.Barrier()
    logger.info("Dynamics calculation completed.")
    # Collect results sequentially on rank 0.
    tag_data, tag_done = 1, 2
    if rank == 0:
        logger.info("Collecting results from all tasks.")
        for result in local_results:
            data.add_data(result)
        remaining = size - 1
        status = MPI.Status()
        while remaining:
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag() == tag_done:
                remaining -= 1
            else:
                data.add_data(msg)
    else:
        for result in local_results:
            comm.send(result, dest=0, tag=tag_data)
        comm.send(None, dest=0, tag=tag_done)
    logger.info("Simulation complete.")
    if progress_enabled:
        batch_bars.close()
    # Collect logs from all ranks and attach combined output on root rank.
    gathered_logs = comm.gather(get_log_output(), root=0)
    if rank == 0:
        data.log = "".join(log for log in gathered_logs if log)
    return data
