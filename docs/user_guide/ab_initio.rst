.. _ab-initio:

==========================
Ab Initio Dynamics
==========================

QC Lab ships with two algorithms tailored to models defined in an
adiabatic basis: :class:`~qclab.algorithms.MeanFieldAbInitio` and
:class:`~qclab.algorithms.FewestSwitchesSurfaceHoppingAbInitio`. Together
with the :class:`~qclab.models.AbInitio` model, these algorithms make
it possible to run on-the-fly nonadiabatic molecular dynamics where
energies, gradients, and derivative couplings are produced by an
external electronic-structure code at every timestep.

This page describes the workflow at a high level. For the underlying
adiabatic-basis machinery (gauge fixing, ``adb_connection``, etc.),
also read :ref:`Algorithms <algorithm>` and :ref:`Coordinates
<coordinates>`.

When to use the adiabatic-basis algorithms
==========================================

Use ``MeanFieldAbInitio`` or ``FewestSwitchesSurfaceHoppingAbInitio``
when:

- there is no global diabatic basis available (typical of an *ab initio*
  electronic-structure calculation), and
- the Hamiltonian and derivative couplings are produced trajectory by
  trajectory by an external solver.

For model problems with a known diabatic basis (spin-boson,
Holstein, FMO, Tully I/II/III), the ordinary diabatic algorithms
:class:`~qclab.algorithms.MeanField` and
:class:`~qclab.algorithms.FewestSwitchesSurfaceHopping` are simpler and
faster. Mixing the two is not supported: a diabatic algorithm cannot run
on an ``AbInitio`` model, and an adiabatic algorithm cannot run on a
diabatic model.

----

The minimum-viable workflow
===========================

The following pseudo-code shows the structure of an ab initio QC Lab
script. It assumes Q-Chem is installed and discoverable on the system
``PATH``.

.. code-block:: python

    import numpy as np
    from qclab import Simulation
    from qclab.models import AbInitio
    from qclab.algorithms import FewestSwitchesSurfaceHoppingAbInitio
    from qclab.dynamics import serial_driver

    # 1. Build the AbInitio model with the molecule's geometry,
    #    masses, normal modes, and Q-Chem keyword arguments.
    sim = Simulation({
        "tmax":        100.0,
        "dt_update":     1.0,
        "dt_collect":   10.0,
        "num_trajs":    20,
        "batch_size":    5,
        "progress_bar": False,
    })
    sim.model = AbInitio({
        "atom_positions":   atom_positions_bohr,
        "atom_masses":      atom_masses_emass,
        "atom_names":       ["C", "H", "H", ...],
        "normal_mode":      normal_modes,           # (3*num_atoms, num_modes)
        "harmonic_frequency": harmonic_frequencies, # (num_modes,) Hartree
        "num_quantum_states": 4,
        "energy_offset":    -100.0,
        "calculator_args": {
            "method_es":   "tddft",
            "exchange":    "B3LYP",
            "basis":       "6-31G*",
            "cis_n_roots": 6,
        },
    })

    # 2. Attach the adiabatic-basis algorithm.
    sim.algorithm = FewestSwitchesSurfaceHoppingAbInitio()

    # 3. Set the initial adiabatic wavefunction (e.g., excited on S2).
    wf0 = np.zeros(sim.model.constants.num_quantum_states, dtype=complex)
    wf0[2] = 1.0
    sim.initial_state["wf_db"] = wf0

    data = serial_driver(sim)

The ``calculator_args`` dictionary is forwarded to the calculator
constructor (:class:`qclab.interfaces.QCLabQChemInterface` by default).
``num_quantum_states`` is the number of electronic states tracked by the
algorithm; it must not exceed the number of states the calculator is
configured to compute (``cis_n_roots`` for TDDFT, etc.).

----

The ``update_ab_initio_property`` task
=======================================

The bridge between the algorithm and the electronic-structure solver is
a single task,
:func:`qclab.tasks.update_tasks.update_ab_initio_property`. It is called
several times per step in both ab initio recipes.

The task takes a ``property_dict`` keyword argument that specifies which
ab initio quantities to compute on this call and which arguments to pass
through to the calculator. The supported properties are:

``energy``
    Computes excited-state energies of the configured number of states.
    Args:

    - ``z``: state-dict key holding the current classical coordinate.
    - ``excited_amplitudes`` (bool): whether to also return the CI / TDDFT
      amplitudes (used internally to track the adiabatic connection
      between consecutive timesteps).

``gradient``
    Computes nuclear gradients of one or more electronic states. Args:

    - ``z``: state-dict key for the current classical coordinate.
    - ``state_inds_gradient``: ``None`` to compute the gradient of
      every state, or the name of a state-dict key holding the indices
      of the states whose gradient is needed (e.g.
      ``"act_surf_ind"`` to compute only the active-surface gradient).

``derivative_coupling``
    Computes derivative couplings between specified pairs of electronic
    states. Args:

    - ``z``: state-dict key for the current classical coordinate.
    - ``state_inds_derivative_coupling``: ``None`` to compute every
      coupling, or the name of a state-dict key holding the
      ``(initial, final)`` state pairs.
    - ``calc_property`` (optional): the name of a state-dict boolean
      array — only those trajectories with ``True`` will trigger a new
      calculation. This is the per-trajectory mask that
      ``FewestSwitchesSurfaceHoppingAbInitio`` uses to compute
      derivative couplings only for trajectories that are about to hop.

``wf_overlaps``
    Computes overlaps between the wavefunctions at consecutive timesteps,
    used by ``MeanFieldAbInitio`` and (optionally) by
    ``FewestSwitchesSurfaceHoppingAbInitio`` to fix the gauge of the
    adiabatic basis. Args:

    - ``z`` and ``z_previous``: state-dict keys for the current and
      previous classical coordinates.
    - ``amplitudes_previous`` and ``amplitudes_current``: state-dict
      keys for the CI / TDDFT amplitudes at the two timesteps.

Each call to ``update_ab_initio_property`` populates
``parameters["ab_initio_property"]`` (a list of per-trajectory dicts)
*and* writes vectorized copies into ``state["aip_<property>"]`` (e.g.
``state["aip_energy"]``, ``state["aip_gradient"]``).

The ingredients that compute ``h_qc``, ``dh_qc_dzc``, and
``derivative_coupling_dzc`` on the ``AbInitio`` model read these cached
results when available and fall back to a synchronous calculator call
otherwise.

----

Algorithm settings
==================

The ab initio variants of the algorithms expose two extra settings on
``algorithm.settings`` (in addition to the ones documented for
:class:`~qclab.algorithms.MeanField` and
:class:`~qclab.algorithms.FewestSwitchesSurfaceHopping`):

``update_wf_adb_eig_num_substeps``
    Number of substeps used inside :func:`update_wf_adb_hop_prob` when
    propagating the adiabatic wavefunction. A larger value reduces
    discretization error in the adiabatic-basis propagation at the cost
    of more linear-algebra work per step. Default: ``10``.

``use_wf_overlaps_for_adb_connection``
    Whether to use overlap-based gauge fixing (computed by the calculator
    via ``wf_overlaps``) rather than coordinate-based gauge fixing
    inside :func:`update_adb_connection`. Default: ``False`` for
    ``MeanFieldAbInitio``, ``True`` for
    ``FewestSwitchesSurfaceHoppingAbInitio``.

----

The Q-Chem interface
====================

The shipped electronic-structure backend lives at
:class:`qclab.interfaces.QCLabQChemInterface`. It writes Q-Chem input
files into a per-trajectory scratch folder, invokes the ``qchem`` binary
via ``subprocess``, and parses the output for energies, gradients,
derivative couplings, normal-mode frequencies, and wavefunction
overlaps.

The interface is selected automatically by the
:class:`~qclab.models.AbInitio` model (its ingredients list registers
``ingredients.ab_initio_property_calculator_qchem`` under the
``ab_initio_property_calculator`` slot). To swap in a different
electronic-structure code, override that slot on the model with a
calculator that exposes the same property names (``energy``,
``gradient``, ``derivative_coupling``, ``wf_overlaps``).

The Q-Chem interface depends on the optional ``ase`` package being
available; if ASE is missing, the ``AbInitio`` model is not exported from
:mod:`qclab.models`. See :ref:`Installing QC Lab <install>` for details.

----

Examples
========

The ``examples/`` folder of the repository contains an MPI-parallel
example script that exercises the diabatic algorithms; an analogous
ab initio example is recommended whenever the Q-Chem environment is
available. The reference
implementations of the ab initio algorithms can be inspected directly
at ``src/qclab/algorithms/mean_field.py`` and
``src/qclab/algorithms/fewest_switches_surface_hopping.py`` (see
:ref:`Ab Initio Surface Hopping Example <ab_initio_fssh_source>`).
