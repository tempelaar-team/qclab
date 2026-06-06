.. _outputs:

==========================================
Recording and Post-processing Outputs
==========================================

QC Lab records simulation results by carrying them through the
``state["output_dict"]`` slot to the Data object returned by the
Dynamics Driver. This section traces that pipeline end-to-end, lists
the standard keys that the built-in Algorithms produce, walks through
adding a new recorded quantity, and shows the save / load and plotting
patterns used to work with a Data object after the run.

For the per-batch lifetime of ``output_dict`` and the
update-Task-plus-collect-Task pair that puts a value there, see
:ref:`State and Parameters Objects <state-and-parameters>`. For the
Data object's class reference, see :ref:`Data <data>`.

----

The output pipeline at a glance
===============================

A single value travels through four stages on its way to a saved
Data object:

#. **Update Task** — runs on every update time step (or only on
   collect time steps when efficiency matters). Computes the
   quantity from State entries and writes it back into the State
   object.
#. **Collect Task** — runs on every collect time step. Copies the
   quantity from the State object into ``state["output_dict"]``
   under its output key.
#. **Driver** — at the same collect time step, calls
   :meth:`Data.add_output_to_data_dict <qclab.data.Data.add_output_to_data_dict>`,
   which sums the entry across the batch axis and divides by the
   running ``norm_factor`` (the total number of trajectories that
   have contributed so far).
#. **Data object** — stores the trajectory-averaged result at the
   correct collect-time index in ``data.data_dict[<key>]``.

The averaging step is fixed: the leading axis (the batch axis) is
summed and divided by the global normalization. Standard deviations
or quantiles across trajectories are **not** kept by the Data object;
recording a per-trajectory quantity is possible (see below) but the
default machinery is set up for trajectory-averaged means.

----

Standard output keys
====================

The built-in Algorithms write the following keys into the Data
object. The list is comprehensive for the default Recipes shipping
with QC Lab.

.. list-table::
   :header-rows: 1
   :widths: 24 18 28 30

   * - Key
     - Shape
     - Dtype
     - Produced by
   * - ``t``
     - ``(T,)``
     - ``float64``
     - every Algorithm (``collect_t``)
   * - ``dm_db``
     - ``(T, N, N)``
     - ``complex128``
     - mean-field, FSSH (diabatic basis)
   * - ``dm_adb``
     - ``(T, N, N)``
     - ``complex128``
     - mean-field (*ab initio*), FSSH (*ab initio*)
   * - ``classical_energy``
     - ``(T,)``
     - ``float64``
     - every Algorithm
   * - ``quantum_energy``
     - ``(T,)``
     - ``float64``
     - every Algorithm
   * - ``seed``
     - ``(num_trajs,)``
     - ``int64``
     - every Driver

Here ``T = len(sim.settings.t_collect)`` is the number of collect
time steps and ``N = sim.model.constants.num_quantum_states``.

The keys ``norm_factor`` and ``seed`` are also present in the Data
object but are bookkeeping rather than recorded physics:
``norm_factor`` is the running normalization used by
``add_output_to_data_dict`` and ``seed`` is the union of all
trajectory seeds processed so far.

----

Adding a new recorded quantity
==============================

The general recipe for adding a quantity is:

#. Write an update Task that computes the quantity and stores it on
   the State object under a chosen key.
#. Write a collect Task that copies that State entry into
   ``state["output_dict"]``.
#. Append the update Task to ``algorithm.update_recipe`` *and* to
   ``algorithm.initialization_recipe`` (the second is so the
   quantity exists at ``t = 0`` before the first collect time step
   runs).
#. Append the collect Task to ``algorithm.collect_recipe``.

A worked example: record the bath kinetic energy
:math:`\tfrac{1}{2 m} \sum_\xi p_\xi^2` alongside the existing
``classical_energy``.

.. code-block:: python

    import numpy as np
    from qclab.functions import z_to_p

    def update_bath_kinetic(sim, state, parameters,
                            *, z_name="z",
                            kinetic_name="bath_kinetic"):
        """Compute the bath kinetic energy from z."""
        m = sim.model.constants.classical_coordinate_mass[np.newaxis, :]
        h = sim.model.constants.classical_coordinate_weight[np.newaxis, :]
        p = z_to_p(state[z_name], m, h)
        state[kinetic_name] = np.sum(p * p / (2.0 * m), axis=1).real
        return state, parameters


    def collect_bath_kinetic(sim, state, parameters,
                             *, kinetic_name="bath_kinetic",
                             kinetic_output_name="bath_kinetic"):
        """Copy the bath kinetic energy into output_dict."""
        state["output_dict"][kinetic_output_name] = state[kinetic_name]
        return state, parameters

Wiring the Tasks into the Algorithm after instantiation uses the
rebinding pattern from :ref:`Recipes <recipe>`:

.. code-block:: python

    sim.algorithm.initialization_recipe = (
        sim.algorithm.initialization_recipe + [update_bath_kinetic]
    )
    sim.algorithm.update_recipe = (
        sim.algorithm.update_recipe + [update_bath_kinetic]
    )
    sim.algorithm.collect_recipe = (
        sim.algorithm.collect_recipe + [collect_bath_kinetic]
    )

    data = serial_driver(sim)

After the run, ``data.data_dict["bath_kinetic"]`` is an array of shape
``(T,)`` holding the trajectory-averaged bath kinetic energy at each
collect time step.

The ``*_name`` keyword arguments follow the convention described in
:ref:`State and Parameters Objects <state-and-parameters>`. The Tasks
above are written so that the State key and the output key are
configurable; this is what lets the same Task be reused under
different names by a later wrapping in :func:`functools.partial`.

----

Saving and loading a Data object
================================

The Data object provides :meth:`save <qclab.data.Data.save>` and
:meth:`load <qclab.data.Data.load>` for round-tripping to disk. The
serialization format is HDF5 (via ``h5py``) by default; if ``h5py``
is unavailable or disabled, the methods fall back to NumPy's
``.npz`` archive.

.. code-block:: python

    data = serial_driver(sim)
    data.save("run_0001.h5")

    # Later, in a separate session:
    from qclab import Data
    data2 = Data()
    data2.load("run_0001.h5")

Two properties of :meth:`load` are worth noting:

- :meth:`load` is **additive**. It calls
  :meth:`add_data <qclab.data.Data.add_data>` internally, so loading
  a file into a Data object that already contains data merges the
  two. To load a file in isolation, construct a fresh Data object
  first (``Data()``).
- Every value stored in ``data_dict`` must be convertible to a NumPy
  array of supported dtype. Nested dictionaries are written as HDF5
  groups; lists of mixed-type values are not supported.

The ``data.log`` string is round-tripped alongside ``data_dict`` —
HDF5 stores it as a file-level attribute, ``.npz`` stores it under
the key ``log``.

----

Plotting recipes
================

The following recipes assume a Data object ``data`` produced by a
diabatic Algorithm such as ``MeanField`` or
``FewestSwitchesSurfaceHopping``. For *ab initio* runs, replace
``dm_db`` with ``dm_adb``.

Diabatic populations
--------------------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    t = data.data_dict["t"]
    dm = data.data_dict["dm_db"]
    populations = np.real(np.einsum("tii->ti", dm))

    plt.plot(t, populations)
    plt.xlabel("time")
    plt.ylabel("diabatic populations")
    plt.show()

Diabatic coherences
-------------------

.. code-block:: python

    coherence = dm[:, 0, 1]
    plt.plot(t, coherence.real, label="Re")
    plt.plot(t, coherence.imag, label="Im")
    plt.xlabel("time")
    plt.ylabel(r"$\rho_{01}$")
    plt.legend()
    plt.show()

Energy conservation check
-------------------------

.. code-block:: python

    e_q = data.data_dict["quantum_energy"]
    e_c = data.data_dict["classical_energy"]
    plt.plot(t, e_q, label="quantum")
    plt.plot(t, e_c, label="classical")
    plt.plot(t, e_q + e_c, label="total")
    plt.xlabel("time")
    plt.ylabel("energy")
    plt.legend()
    plt.show()

For mean-field dynamics the total energy is conserved to integrator
accuracy; for FSSH the energy is conserved trajectory-wise at hops
but the trajectory-averaged total may drift slightly when forbidden
hops occur.

----

Recording per-trajectory quantities
===================================

The default averaging step in
:meth:`add_output_to_data_dict <qclab.data.Data.add_output_to_data_dict>`
sums along the batch axis, so the per-trajectory information is
discarded by the time the Data object is written. To keep the
per-trajectory record, save the State entry under a different shape
that includes the trajectory index, for example by writing one entry
per trajectory under a per-seed key. A simpler alternative is to use
``serial_driver`` with ``batch_size = 1`` and ``num_trajs = 1``
inside a loop; each call returns a Data object that holds a single
trajectory's averaged-but-not-summed values.

The recommended pattern when per-trajectory output is needed across
many trajectories is to record into a separate file (e.g., NumPy
``.npy`` per trajectory inside the update Task) and aggregate
externally after the run.
