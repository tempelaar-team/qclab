.. _conventions:

==========================
Conventions Reference
==========================

This section lists the standard names used in QC Lab. It is intended as
a reference for users who need to know:

- the slot name for an ingredient,
- the key under which a quantity lives in the State object or the
  Parameters object,
- the constant a built-in ingredient reads from a Model object's
  Constants object, or
- the setting an Algorithm object understands.

Throughout this section we use the shorthand:

- ``B = sim.settings.batch_size`` — the number of trajectories carried in
  a single batch.
- ``C = num_classical_coordinates`` — the number of classical coordinates
  in the model.
- ``N = num_quantum_states`` — the number of quantum states in the model.

The first axis of every State-object array is the batch axis.

----

Standard ingredient slots
=========================

These are the names that algorithms use when calling
``sim.model.get(...)``. The first element of each ``(name, callable)``
tuple in a Model object's ``ingredients`` list must match one of these
slot names (or be an ``_init_*`` initializer). The signature of every
ingredient is ``f(model, parameters, **kwargs)``. The list below is
comprehensive.

.. list-table::
   :header-rows: 1
   :widths: 20 25 30 25

   * - Slot
     - Required kwargs
     - Returns
     - Used by
   * - ``h_q``
     - ``batch_size``
     - ``(B, N, N)`` complex Hamiltonian
     - every algorithm
   * - ``h_qc``
     - ``z``
     - ``(B, N, N)`` complex Hamiltonian
     - every algorithm
   * - ``h_c``
     - ``z``
     - ``(B,)`` real classical energy
     - mean-field, FSSH
   * - ``dh_qc_dzc``
     - ``z``
     - sparse ``(inds, mels, shape)`` for ``(B, C, N, N)``
     - every algorithm; falls back to finite differences if absent
   * - ``dh_c_dzc``
     - ``z``
     - ``(B, C)`` complex gradient
     - every algorithm; falls back to finite differences if absent
   * - ``init_classical``
     - ``seed``
     - ``(B, C)`` complex initial coordinates
     - every algorithm; falls back to MCMC if absent
   * - ``hop``
     - ``z``, ``resc_dir_z``, ``eigval_diff``
     - ``(shift, hop_bool)``
     - FSSH only
   * - ``derivative_coupling_dzc``
     - ``z``
     - ``(B, C, N, N)`` complex
     - ab initio only
   * - ``gauge_field_force``
     - ``z``, ``state_ind``
     - ``(B, C)`` complex
     - optional, when ``use_gauge_field_force == True``
   * - ``ab_initio_property_calculator``
     - ``property_dict``, ``traj_ind``
     - dict of energies / gradients / couplings
     - ab initio only

``_init_*`` initializer ingredients
-----------------------------------

Names that begin with an underscore are initializers. They are called by
:meth:`Model.initialize_constants <qclab.Model.initialize_constants>`
whenever a constant changes, and their purpose is to derive internal
constants from the user-facing ones. The four initializer names used by
the built-in models are:

- ``_init_model`` — sets sizes (``num_quantum_states``,
  ``num_classical_coordinates``) and per-coordinate metadata
  (``classical_coordinate_mass``, ``classical_coordinate_weight``).
- ``_init_h_q`` — derives constants for the ``h_q`` ingredient.
- ``_init_h_qc`` — derives constants for the ``h_qc`` ingredient.
- ``_init_h_c`` — derives constants for the ``h_c`` ingredient.

A Model object can introduce additional ``_init_*`` initializers if
needed.

----

Standard State-object keys
==========================

Keys follow ``lower_snake_case``. Each key listed below is read or
written by at least one of the built-in tasks; reusing these names in a
custom task removes the need to rebind keys when inserting the task into
an existing recipe. The categories below cover the keys used by the
built-in tasks. Custom tasks may introduce additional keys.

Trajectory bookkeeping
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Key
     - Shape / type
     - Meaning
   * - ``seed``
     - ``(B,)`` int
     - Per-trajectory random seed
   * - ``branch_ind``
     - ``(B,)`` int
     - FSSH branch index (deterministic mode)
   * - ``t``
     - ``(B,)`` float64
     - Current time
   * - ``output_dict``
     - dict
     - Values to collect this step
   * - ``norm_factor``
     - scalar
     - Trajectory-average normalization (= ``batch_size``)

Classical coordinates and RK4 intermediates
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Key
     - Shape / type
     - Meaning
   * - ``z``
     - ``(B, C)`` complex128
     - Current classical coordinate
   * - ``z_1``, ``z_2``, ``z_3``
     - ``(B, C)`` complex128
     - RK4 sub-step intermediates
   * - ``z_previous``
     - ``(B, C)`` complex128
     - Previous-timestep ``z``
   * - ``z_rk4_k1``, ``z_rk4_k2``, ``z_rk4_k3``
     - ``(B, C)`` complex128
     - RK4 slopes

Hamiltonian matrices
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Key
     - Shape / type
     - Meaning
   * - ``h_q``
     - ``(B, N, N)`` complex128
     - Quantum Hamiltonian
   * - ``h_qc``
     - ``(B, N, N)`` complex128
     - Quantum-classical Hamiltonian
   * - ``h_q_tot``
     - ``(B, N, N)`` complex128
     - ``h_q + h_qc``
   * - ``h_q_tot_previous``
     - ``(B, N, N)`` complex128
     - Previous-step value

Forces
------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Key
     - Shape / type
     - Meaning
   * - ``classical_force``
     - ``(B, C)`` complex128
     - Force from ``dh_c_dzc``
   * - ``quantum_classical_force``
     - ``(B, C)`` complex128
     - Force from :math:`\langle \psi | \partial_{z^*} h_{qc} | \psi \rangle`
   * - ``*_force_previous``
     - same as above
     - Previous-step value

Diagonalization output
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Key
     - Shape / type
     - Meaning
   * - ``eigvals``
     - ``(B, N)`` float64
     - Eigenvalues of ``h_q_tot``
   * - ``eigvecs``
     - ``(B, N, N)`` complex128
     - Eigenvectors (columns are states)
   * - ``eigvecs_previous``
     - ``(B, N, N)`` complex128
     - Previous-step eigenvectors

Wavefunctions
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Key
     - Shape / type
     - Meaning
   * - ``wf_db``
     - ``(B, N)`` complex128
     - Wavefunction in diabatic basis
   * - ``wf_adb``
     - ``(B, N)`` complex128
     - Wavefunction in adiabatic basis
   * - ``act_surf_wf``
     - ``(B, N)`` complex128
     - Active-surface unit vector (FSSH)

FSSH-specific
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Key
     - Shape / type
     - Meaning
   * - ``act_surf``
     - ``(B, N)`` int
     - One-hot active surface
   * - ``act_surf_ind``
     - ``(B,)`` int
     - Active-surface index
   * - ``act_surf_ind_0``
     - ``(B,)`` int
     - Initial active surface
   * - ``hop_prob``
     - ``(B, N)`` float64
     - Hopping probabilities
   * - ``hop_ind``
     - ``(H,)`` int
     - Indices of trajectories attempting a hop
   * - ``hop_dest``
     - ``(H,)`` int
     - Destination surfaces
   * - ``hop_bool``
     - ``(B,)`` bool
     - Whether each trajectory hops
   * - ``hop_pairs``
     - ``(B, 2)`` int
     - ``(initial, final)`` state pairs
   * - ``hop_successful``
     - ``(B,)`` bool
     - Whether the (already-attempted) hop succeeded
   * - ``dm_adb_0``
     - ``(B, N, N)`` complex128
     - Initial adiabatic density matrix

Density matrices and energies
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Key
     - Shape / type
     - Meaning
   * - ``dm_db``
     - ``(B, N, N)`` complex128
     - Diabatic density matrix
   * - ``dm_adb``
     - ``(B, N, N)`` complex128
     - Adiabatic density matrix
   * - ``classical_energy``
     - ``(B,)`` float64
     - Per-trajectory classical energy
   * - ``quantum_energy``
     - ``(B,)`` float64
     - Per-trajectory quantum energy

Ab initio extras
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Key
     - Shape / type
     - Meaning
   * - ``aip_excited_amplitudes``
     - varies
     - From the ab initio property calculator
   * - ``derivative_coupling_dzc``
     - ``(B, C, N, N)`` complex128
     - Derivative coupling tensor
   * - ``adb_connection``
     - ``(B, N, N)`` complex128
     - Adiabatic connection matrix

Suffix conventions
------------------

- ``_previous`` — value from the prior time step.
- ``_0`` — initial-time reference (e.g. ``dm_adb_0``, ``act_surf_ind_0``).
- ``_ind`` — integer index (e.g. ``act_surf_ind``, ``traj_ind``).
- ``_name`` — only used for task keyword arguments whose value is a
  string key name (e.g. ``z_name="z"``).

----

Standard Model-object constants
===============================

The names below are used by the built-in Model objects and ingredients
in QC Lab. They are listed by category. A custom Model object may
introduce additional constants of its own; the entries here are
representative of the conventions used by the built-in code.

Sizes (always ``num_`` prefix)
------------------------------

- ``num_quantum_states``
- ``num_classical_coordinates``
- ``num_atoms``

Per-coordinate metadata
-----------------------

- ``classical_coordinate_mass`` ``(C,)``
- ``classical_coordinate_weight`` ``(C,)``
- ``harmonic_frequency`` ``(C,)``

Initial conditions (``init_`` prefix)
-------------------------------------

- ``init_position``
- ``init_momentum``

Coupling constants (named after the consuming ingredient)
---------------------------------------------------------

- ``diagonal_linear_coupling`` — used by ``h_qc_diagonal_linear``.
- ``nearest_neighbor_hopping_energy``, ``nearest_neighbor_periodic`` —
  used by ``h_q_nearest_neighbor``.
- ``two_level_00``, ``two_level_11``, ``two_level_01_re``,
  ``two_level_01_im`` — used by ``h_q_two_level``.
- ``coherent_state_displacement`` — used by
  ``init_classical_wigner_coherent_state``.

Atomistic and ab initio constants
---------------------------------

- ``atom_names``
- ``atom_masses``
- ``atom_positions``
- ``normal_mode``
- ``energy_offset``
- ``calculator_args``

Numerical tuning knobs (``<consumer>_<knob>``)
----------------------------------------------

- ``numerical_fssh_hop_gamma_range``
- ``numerical_fssh_hop_max_iter``
- ``numerical_fssh_hop_num_points``
- ``numerical_fssh_hop_threshold``
- ``dh_c_dzc_finite_difference_delta``

User-facing physical constants (conventional symbols)
-----------------------------------------------------

- ``kBT``, ``V``, ``E``, ``A``, ``W``, ``J``, ``N``, ``g``, ``w``,
  ``l_reorg``, ``w_c``

----

Standard Algorithm-object settings
==================================

The settings below are the ones recognized by the built-in Algorithm
objects. The list is comprehensive for the built-in algorithms; a
custom Algorithm object may introduce additional settings of its own.

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Setting
     - Type
     - Default
     - Used by
   * - ``tmax``
     - float
     - 10.0
     - every simulation
   * - ``dt_update``
     - float
     - 0.001
     - every simulation
   * - ``dt_collect``
     - float
     - 0.1
     - every simulation
   * - ``num_trajs``
     - int
     - 100
     - every simulation
   * - ``batch_size``
     - int
     - 25
     - every simulation
   * - ``progress_bar``
     - bool
     - True
     - every simulation
   * - ``debug``
     - bool
     - False
     - gates expensive sanity checks
   * - ``fssh_deterministic``
     - bool
     - False
     - FSSH
   * - ``gauge_fixing``
     - str
     - ``"sign_overlap"``
     - FSSH
   * - ``use_gauge_field_force``
     - bool
     - False
     - FSSH
   * - ``update_wf_adb_eig_num_substeps``
     - int
     - 10
     - ab initio
   * - ``use_wf_overlaps_for_adb_connection``
     - bool
     - varies
     - ab initio

Boolean flags start with ``use_`` or ``is_``; mode strings are descriptive
``snake_case``.

.. note::

    The ``"phase_der_couple"`` value of ``gauge_fixing`` is only required
    when the Hamiltonian or derivative couplings are complex-valued
    (e.g. models with magnetic fields or complex hopping). For
    real-valued problems such as the Tully models or the standard
    spin-boson model, the default ``"sign_overlap"`` is sufficient and
    avoids unnecessary overhead.

----

Local variable names in physics code
====================================

The following short names appear in the bodies of the built-in
ingredients and tasks. Reusing them in custom code keeps new functions
visually consistent with the existing ones. The list is representative
of the conventions used in the built-in code.

- coordinates: ``z``, ``q``, ``p``
- per-coordinate quantities: ``m`` (mass), ``h`` (weight),
  ``w`` (frequency)
- thermal energy: ``kBT``
- eigenpairs: ``evec_i``, ``evec_j``, ``eval_i``, ``eval_j``,
  ``eigval_diff``
- sparse triple (always in this order): ``inds``, ``mels``, ``shape``
- sizes: ``batch_size``, ``num_classical_coordinates``,
  ``num_quantum_states``
- indices: ``traj_ind``, ``state_ind``, ``act_surf_ind``, ``t_ind``
