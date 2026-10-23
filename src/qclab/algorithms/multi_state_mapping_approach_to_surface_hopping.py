"""
This module contains the MultiStateMappingApproachToSurfaceHopping algorithm class.
"""

from functools import partial
from qclab.algorithm import Algorithm
from qclab import tasks


class MultiStateMappingApproachToSurfaceHopping(Algorithm):
    """
    Multi-state mapping approach to surface hopping algorithm class.

    Implements the multi-state MASH of Runeson and Manolopoulos (J. Chem. Phys. 159, 094115,
    2023).

    Johan Runeson is acknowledged for reviewing this implementation.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        self.default_settings = {
            "gauge_fixing": "sign_overlap",
            "use_gauge_field_force": False,
            # This reversal is not representation-invariant; set False for basis invariance.
            "reverse_frustrated_hops": True,
            "msmash_max_hops": 10,
            "msmash_num_bisections": 10,
        }
        super().__init__(self.default_settings, settings)

    initialization_recipe = [
        tasks.initialize_variable_objects,
        tasks.initialize_z,
        tasks.initialize_wf_db_msmash,
        tasks.update_h_q_tot,
        partial(
            tasks.diagonalize_matrix,
            matrix_name="h_q_tot",
            eigvals_name="eigvals",
            eigvecs_name="eigvecs",
        ),
        partial(
            tasks.update_eigvecs_gauge,
            gauge_fixing="phase_der_couple",
            eigvecs_previous_name="eigvecs",
            output_eigvecs_name="eigvecs",
        ),
        partial(tasks.copy_in_state, copy_name="eigvecs_previous", orig_name="eigvecs"),
        tasks.initialize_act_surf_msmash,
        tasks.update_act_surf_wf,
        tasks.initialize_norm_factor,
    ]

    update_recipe = [
        # Record the pre-step snapshot replayed by the hop loop for any surface that crosses.
        partial(tasks.copy_in_state, copy_name="z_previous", orig_name="z"),
        partial(tasks.copy_in_state, copy_name="wf_db_previous", orig_name="wf_db"),
        partial(
            tasks.copy_in_state,
            copy_name="act_surf_ind_previous",
            orig_name="act_surf_ind",
        ),
        # h_q_tot, eigvals, eigvecs and act_surf_wf are consistent with z from the end of
        # the previous step (or initialization), so no start-of-step diagonalization.
        # (Dead start-block QC force removed: the wf_changed=True call below recomputes it at the
        # same z before k1 consumes it.)
        tasks.update_classical_force,
        # Begin RK4 step: classical z by RK4 under the active-surface force, electronic wf_db by
        # the exact propagator. z-native (no q/p split), so representation-covariant to
        # integrator order. Composes the shared qclab RK4 Tasks.
        partial(
            tasks.update_quantum_classical_force,
            wf_db_name="act_surf_wf",
            wf_changed=True,
        ),
        tasks.update_z_rk4_k123,
        partial(tasks.update_classical_force, z_name="z_1"),
        partial(
            tasks.update_quantum_classical_force,
            wf_db_name="act_surf_wf",
            z_name="z_1",
            wf_changed=False,
        ),
        partial(tasks.update_z_rk4_k123, z_name="z", z_k_name="z_2", k_name="z_rk4_k2"),
        partial(tasks.update_classical_force, z_name="z_2"),
        partial(
            tasks.update_quantum_classical_force,
            wf_db_name="act_surf_wf",
            z_name="z_2",
            wf_changed=False,
        ),
        partial(
            tasks.update_z_rk4_k123,
            z_name="z",
            z_k_name="z_3",
            k_name="z_rk4_k3",
            dt_factor=1.0,
        ),
        partial(tasks.update_classical_force, z_name="z_3"),
        partial(
            tasks.update_quantum_classical_force,
            wf_db_name="act_surf_wf",
            z_name="z_3",
            wf_changed=False,
        ),
        tasks.update_z_rk4_k4,
        tasks.update_wf_db_propagator,
        # Re-diagonalize and recompute the active-surface force at the new z.
        partial(tasks.copy_in_state, copy_name="eigvecs_previous", orig_name="eigvecs"),
        tasks.update_h_q_tot,
        partial(
            tasks.diagonalize_matrix,
            matrix_name="h_q_tot",
            eigvals_name="eigvals",
            eigvecs_name="eigvecs",
        ),
        partial(tasks.update_eigvecs_gauge, output_eigvecs_name="eigvecs"),
        tasks.update_act_surf_wf,
        # End RK4 step.
        tasks.update_resolve_hops_msmash,
    ]

    collect_recipe = [
        tasks.update_t,
        tasks.update_dm_db_msmash,
        tasks.collect_t,
        tasks.collect_dm_db,
    ]
