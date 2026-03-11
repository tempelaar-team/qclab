"""
This module contains the dynamics core.
"""

import numpy as np
from tqdm import tqdm


def run_dynamics(sim, state, parameters, data):
    """
    Dynamics core for QC Lab.

    .. rubric:: Args

    sim: Simulation
        The Simulation object containing the model, algorithm, and settings.
    state: dict
        The State object containing the simulation seeds.
    parameters: dict
        The Parameters object containing any additional parameters.
    data: Data
        The Data object for collecting output data.

    .. rubric:: Returns

    data: Data
        The updated Data object containing collected output data.
    """
    # Define an update iterator using tqdm if progress_bar is True.
    t_update_iterator = sim.settings.t_update_n
    if getattr(sim.settings, "progress_bar", True):
        t_update_iterator = tqdm(t_update_iterator)

    # Iterate over each time step.
    for sim.t_ind in t_update_iterator:
        if sim.t_ind == 0:
            # Execute initialization recipe.
            state, parameters = sim.algorithm.execute_recipe(
                sim, state, parameters, sim.algorithm.initialization_recipe
            )
        # Detect collect timesteps.
        if np.mod(sim.t_ind, sim.settings.dt_collect_n) == 0:
            # Calculate output variables.
            state, parameters = sim.algorithm.execute_recipe(
                sim, state, parameters, sim.algorithm.collect_recipe
            )
            # Collect totals in output dictionary.
            data.add_output_to_data_dict(sim, state, sim.t_ind)
        # Execute update recipe.
        state, parameters = sim.algorithm.execute_recipe(
            sim, state, parameters, sim.algorithm.update_recipe
        )
    return data
