.. _ab-initio:

==========================
Ab Initio Dynamics
==========================

QC Lab ships with two algorithms tailored to Model objects defined in an
adiabatic basis: :class:`~qclab.algorithms.MeanFieldAbInitio` and
:class:`~qclab.algorithms.FewestSwitchesSurfaceHoppingAbInitio`. Together
with the :class:`~qclab.models.AbInitio` Model object, these algorithms
support on-the-fly nonadiabatic dynamics in which energies, gradients,
and derivative couplings are produced by an external electronic-structure
code at every update time step.

This section gives a high-level overview of the workflow. The
:ref:`Algorithms <algorithm>` and :ref:`Coordinates <coordinates>`
sections cover the underlying adiabatic-basis machinery (gauge fixing,
adiabatic connection, and so on).

When to use the adiabatic-basis algorithms
==========================================

The adiabatic-basis algorithms ``MeanFieldAbInitio`` and
``FewestSwitchesSurfaceHoppingAbInitio`` are used when:

- there is no global diabatic basis available, as is typical of an
  ab initio electronic-structure calculation, and
- the Hamiltonian and derivative couplings are produced trajectory by
  trajectory by an external solver.

For model problems with a known diabatic basis (spin-boson, Holstein,
FMO, Tully I/II/III), the diabatic algorithms
:class:`~qclab.algorithms.MeanField` and
:class:`~qclab.algorithms.FewestSwitchesSurfaceHopping` are more
appropriate. The adiabatic-basis algorithms are not compatible with the
diabatic Model objects, and vice versa.

----

A minimum-viable workflow
=========================

The following template shows the structure of an ab initio QC Lab
script. It assumes that Q-Chem is installed and discoverable on the
system ``PATH``.

.. code-block:: python

    import numpy as np
    from qclab import Simulation
    from qclab.models import AbInitio
    from qclab.algorithms import FewestSwitchesSurfaceHoppingAbInitio
    from qclab.dynamics import serial_driver

    # 1. Build the AbInitio Model object with the geometry, masses,
    #    normal modes, and Q-Chem keyword arguments.
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

    # 2. Attach the adiabatic-basis Algorithm object.
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
configured to compute (``cis_n_roots`` for TDDFT, for example).

----

The ``update_ab_initio_property`` task
=======================================

The bridge between the Algorithm object and the electronic-structure
solver is the task
:func:`qclab.tasks.update_tasks.update_ab_initio_property`. It is
called multiple times per update time step in both ab initio recipes.

The task takes a ``property_dict`` keyword argument that specifies which
ab initio quantities to compute on this call and which State-object
keys to pass through to the calculator. The supported properties are
listed below. The list is comprehensive for the built-in ab initio
algorithms.

``energy``
    Computes excited-state energies of the configured number of states.
    Arguments:

    - ``z``: name of the State-object key holding the current classical
      coordinate.
    - ``excited_amplitudes`` (bool): whether to also return the CI /
      TDDFT amplitudes, used internally to track the adiabatic
      connection between consecutive update time steps.

``gradient``
    Computes nuclear gradients of one or more electronic states.
    Arguments:

    - ``z``: name of the State-object key for the current classical
      coordinate.
    - ``state_inds_gradient``: ``None`` to compute the gradient of
      every state, or the name of a State-object key holding the
      indices of the states whose gradient is needed (e.g.
      ``"act_surf_ind"`` to compute only the active-surface gradient).

``derivative_coupling``
    Computes derivative couplings between specified pairs of electronic
    states. Arguments:

    - ``z``: name of the State-object key for the current classical
      coordinate.
    - ``state_inds_derivative_coupling``: ``None`` to compute every
      coupling, or the name of a State-object key holding the
      ``(initial, final)`` state pairs.
    - ``calc_property`` (optional): the name of a State-object boolean
      array. Only those trajectories with ``True`` trigger a new
      calculation. This is the per-trajectory mask that
      ``FewestSwitchesSurfaceHoppingAbInitio`` uses to compute
      derivative couplings only for trajectories that are about to hop.

``wf_overlaps``
    Computes overlaps between the wavefunctions at consecutive update
    time steps, used by ``MeanFieldAbInitio`` and (optionally) by
    ``FewestSwitchesSurfaceHoppingAbInitio`` to fix the gauge of the
    adiabatic basis. Arguments:

    - ``z`` and ``z_previous``: names of the State-object keys for the
      current and previous classical coordinates.
    - ``amplitudes_previous`` and ``amplitudes_current``: names of the
      State-object keys for the CI / TDDFT amplitudes at the two time
      steps.

Each call to ``update_ab_initio_property`` populates
``parameters["ab_initio_property"]`` (a list of per-trajectory
dictionaries) and writes vectorized copies into
``state["aip_<property>"]`` (for example, ``state["aip_energy"]``,
``state["aip_gradient"]``).

The ingredients that compute ``h_qc``, ``dh_qc_dzc``, and
``derivative_coupling_dzc`` on the :class:`~qclab.models.AbInitio` Model
object read these cached results when available and fall back to a
synchronous calculator call otherwise.

----

Algorithm settings
==================

The ab initio variants of the algorithms recognize two settings in
addition to the ones documented for
:class:`~qclab.algorithms.MeanField` and
:class:`~qclab.algorithms.FewestSwitchesSurfaceHopping`:

``update_wf_adb_eig_num_substeps``
    Number of substeps used inside
    :func:`~qclab.tasks.update_tasks.update_wf_adb_hop_prob` when
    propagating the adiabatic wavefunction. A larger value reduces
    discretization error in the adiabatic-basis propagation at the cost
    of more linear-algebra work per update time step. Default: ``10``.

``use_wf_overlaps_for_adb_connection``
    Whether to use overlap-based gauge fixing (computed by the calculator
    via ``wf_overlaps``) rather than coordinate-based gauge fixing
    inside :func:`~qclab.tasks.update_tasks.update_adb_connection`.
    Default: ``False`` for ``MeanFieldAbInitio``; ``True`` for
    ``FewestSwitchesSurfaceHoppingAbInitio``.

----

The Q-Chem interface
====================

The electronic-structure backend that ships with QC Lab is
:class:`qclab.interfaces.QCLabQChemInterface`. It writes Q-Chem input
files into a per-trajectory scratch folder, invokes the ``qchem``
binary via ``subprocess``, and parses the output for energies,
gradients, derivative couplings, normal-mode frequencies, and
wavefunction overlaps.

The interface is selected by the :class:`~qclab.models.AbInitio` Model
object's ingredient list, which registers
``ingredients.ab_initio_property_calculator_qchem`` under the
``ab_initio_property_calculator`` slot. To use a different
electronic-structure code, override that slot on the Model object with
a calculator that exposes the same property names (``energy``,
``gradient``, ``derivative_coupling``, ``wf_overlaps``).

The Q-Chem interface depends on the optional ``ase`` package. If ASE is
missing, the :class:`~qclab.models.AbInitio` Model object is not
exported from :mod:`qclab.models`. See :ref:`Installing QC Lab
<install>` for details.

----

Examples
========

The ``examples/`` folder of the repository contains an MPI-parallel
example script that uses the diabatic algorithms. The reference
implementations of the ab initio algorithms can be inspected directly
at ``src/qclab/algorithms/mean_field.py`` and
``src/qclab/algorithms/fewest_switches_surface_hopping.py``; the latter
is also reproduced in the :ref:`Ab Initio Surface Hopping Example
<ab_initio_fssh_source>`.
