"""
This module contains the FewestSwitchesSurfaceHopping algorithm class.
"""

from functools import partial
from qclab.algorithm import Algorithm
from qclab import tasks


class FewestSwitchesSurfaceHopping(Algorithm):
    """
    Fewest switches surface hopping algorithm class.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        self.default_settings = {
            "fssh_deterministic": False,
            "gauge_fixing": "sign_overlap",
            "use_gauge_field_force": False,
        }
        super().__init__(self.default_settings, settings)

    initialization_recipe = [
        tasks.initialize_variable_objects,
        tasks.initialize_norm_factor,
        tasks.initialize_branch_seeds,
        tasks.initialize_z,
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
        ),
        partial(tasks.copy_in_state, copy_name="eigvecs_previous", orig_name="eigvecs"),
        partial(
            tasks.update_vector_basis,
            input_vec_name="wf_db",
            basis_name="eigvecs",
            output_vec_name="wf_adb",
            adb_to_db=False,
        ),
        tasks.initialize_random_values_fssh,
        tasks.initialize_active_surface,
        tasks.initialize_dm_adb_0_fssh,
        tasks.update_act_surf_wf,
    ]

    update_recipe = [
        partial(tasks.copy_in_state, copy_name="eigvecs_previous", orig_name="eigvecs"),
        # Begin RK4 integration steps.
        tasks.update_classical_force,
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
        # End RK4 integration steps.
        tasks.update_wf_db_propagator,
        tasks.update_h_q_tot,
        partial(
            tasks.diagonalize_matrix,
            matrix_name="h_q_tot",
            eigvals_name="eigvals",
            eigvecs_name="eigvecs",
        ),
        tasks.update_eigvecs_gauge,
        partial(
            tasks.update_vector_basis,
            input_vec_name="wf_db",
            basis_name="eigvecs",
            output_vec_name="wf_adb",
            adb_to_db=False,
        ),
        tasks.update_hop_prob_fssh,
        tasks.update_hop_inds_fssh,
        tasks.update_hop_vals_fssh,
        tasks.update_z_hop,
        tasks.update_act_surf_hop,
        tasks.update_act_surf_wf,
    ]

    collect_recipe = [
        tasks.update_t,
        tasks.update_dm_db_fssh,
        tasks.update_quantum_energy_act_surf,
        tasks.update_classical_energy_fssh,
        tasks.collect_t,
        tasks.collect_dm_db,
        tasks.collect_quantum_energy,
        tasks.collect_classical_energy,
    ]


class FewestSwitchesSurfaceHoppingAbInitio(Algorithm):
    """
    Fewest switches surface hopping algorithm class implemented in the adiabatic basis
    for compatibility with ab initio calculations.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        self.default_settings = {
            "fssh_deterministic": False,
            "use_gauge_field_force": False,
            "update_wf_adb_eig_num_substeps": 10,
            "use_wf_overlaps_for_adb_connection": True,
        }
        super().__init__(self.default_settings, settings)

    initialization_recipe = [
        tasks.initialize_variable_objects,
        partial(tasks.copy_to_parameters, state_name="seed", parameters_name="seed"),
        tasks.initialize_norm_factor,
        tasks.initialize_branch_seeds,
        tasks.initialize_z,
        partial(
            tasks.update_ab_initio_property,
            property_dict={
                "energy": {"z": "z", "excited_amplitudes": True},
                "gradient": {"z": "z", "state_inds_gradient": None},
                "derivative_coupling": {
                    "z": "z",
                    "state_inds_derivative_coupling": None,
                },
            },
        ),
        tasks.update_h_q_tot,
        tasks.update_classical_force,
        tasks.update_derivative_coupling_dzc,
        partial(tasks.update_quantum_classical_force, wf_db_name="wf_adb"),
        partial(tasks.update_adb_connection, update_derivative_coupling=False),
        tasks.initialize_random_values_fssh,
        tasks.initialize_active_surface,
        tasks.initialize_dm_adb_0_fssh,
        partial(
            tasks.diagonalize_matrix,
            matrix_name="h_q_tot",
            eigvals_name="eigvals",
            eigvecs_name="eigvecs",
        ),
        tasks.update_act_surf_wf,
        tasks.update_quantum_energy_act_surf,
        tasks.update_classical_energy_fssh,
    ]

    update_recipe = [
        partial(
            tasks.copy_in_state,
            copy_name="aip_excited_amplitudes_previous",
            orig_name="aip_excited_amplitudes",
        ),
        partial(
            tasks.copy_in_state,
            copy_name="eigvecs_previous",
            orig_name="eigvecs",
        ),
        partial(
            tasks.copy_in_state,
            copy_name="adb_connection_previous",
            orig_name="adb_connection",
        ),
        partial(tasks.copy_in_state, copy_name="h_q_tot_previous", orig_name="h_q_tot"),
        partial(
            tasks.copy_in_state,
            copy_name="quantum_classical_force_previous",
            orig_name="quantum_classical_force",
        ),
        partial(
            tasks.copy_in_state,
            copy_name="classical_force_previous",
            orig_name="classical_force",
        ),
        partial(
            tasks.copy_in_state,
            copy_name="z_previous",
            orig_name="z",
        ),
        tasks.update_q_velocity_verlet,
        partial(
            tasks.update_ab_initio_property,
            property_dict={
                "energy": {"z": "z", "excited_amplitudes": True},
                "wf_overlaps": {
                    "z": "z",
                    "z_previous": "z_previous",
                    "amplitudes_previous": "aip_excited_amplitudes_previous",
                    "amplitudes_current": "aip_excited_amplitudes",
                },
            },
        ),
        # tasks.update_adb_connection,
        partial(tasks.update_adb_connection, update_derivative_coupling=True),
        tasks.update_h_q_tot,
        partial(
            tasks.diagonalize_matrix,
            matrix_name="h_q_tot",
            eigvals_name="eigvals",
            eigvecs_name="eigvecs",
        ),
        partial(
            tasks.update_wf_adb_hop_prob,
            update_hopping_probabilities=True,
        ),
        partial(
            tasks.update_hop_inds_fssh,
            hop_bool_name="hop_bool",
            hop_pairs_name="hop_pairs",
        ),
        partial(
            tasks.update_ab_initio_property,
            property_dict={
                "derivative_coupling": {
                    "calc_property": "hop_bool",
                    "z": "z",
                    "state_inds_derivative_coupling": "hop_pairs",
                },
            },
        ),
        tasks.update_derivative_coupling_dzc,
        partial(
            tasks.update_hop_vals_fssh,
            derivative_coupling_dzc_name="derivative_coupling_dzc",
        ),
        tasks.update_z_hop,
        tasks.update_act_surf_hop,
        tasks.update_act_surf_wf,
        partial(
            tasks.update_ab_initio_property,
            property_dict={
                "gradient": {"z": "z", "state_inds_gradient": "act_surf_ind"},
            },
        ),
        partial(tasks.update_quantum_classical_force, wf_db_name="act_surf_wf"),
        tasks.update_p_velocity_verlet,
        tasks.update_classical_force,
    ]

    collect_recipe = [
        tasks.update_t,
        tasks.update_dm_db_fssh,
        tasks.update_quantum_energy_act_surf,
        tasks.update_classical_energy_fssh,
        tasks.collect_t,
        partial(tasks.collect_dm_db, dm_db_name="dm_adb", dm_db_output_name="dm_adb"),
        tasks.collect_quantum_energy,
        tasks.collect_classical_energy,
    ]
