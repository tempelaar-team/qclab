.. _architecture:

==========================
Architecture Overview
==========================

A QC Lab simulation has two coequal parts — a Model object and an
Algorithm object — held inside a Simulation object and executed by a
Dynamics Driver. The Driver returns a Data object containing the
trajectory-averaged results. This section names the components, shows
how they fit together, and traces the path a single batch of
trajectories takes from input to output.

----

Components
==========

A QC Lab simulation is described by a small number of objects with
well-defined responsibilities. The following list is comprehensive
for the public API.

.. list-table::
   :header-rows: 1
   :widths: 22 50 28

   * - Object
     - Role
     - Defined in
   * - Simulation object
     - Top-level container. Holds the Model object, the Algorithm
       object, the per-run settings on ``sim.settings``, and the
       ``initial_state`` dictionary that seeds the wavefunction.
     - ``simulation.py``
   * - Model object
     - Physical system. Holds a Constants object on ``model.constants``
       and a list of Ingredients on ``model.ingredients`` that compute
       the Hamiltonians and initial conditions.
     - ``model.py``
   * - Algorithm object
     - Dynamics method. Holds three Recipes —
       ``initialization_recipe``, ``update_recipe``, and
       ``collect_recipe`` — together with algorithm-specific settings
       on ``algorithm.settings``.
     - ``algorithm.py``
   * - Constants object
     - Attribute holder with an optional update-on-write callback. Used
       on ``sim.settings``, ``model.constants``, and
       ``algorithm.settings``; the callback fires only on
       ``model.constants`` and re-runs the Model's ``_init_*``
       Ingredients.
     - ``constants.py``
   * - Data object
     - Trajectory-averaged output container. Stores results in
       ``data.data_dict``, captures the in-memory log, and round-trips
       to HDF5 or ``.npz``.
     - ``data.py``

Two further dictionaries are created and discarded once per batch by
the Dynamics Driver:

- The **State object** carries the per-trajectory quantities that
  change during the simulation (the wavefunction, the complex-classical
  coordinate, eigenvectors, the active surface in the FSSH case).
- The **Parameters object** carries the auxiliary quantities computed
  once and reused across update time steps (for example the cached
  output of an electronic-structure call in the *ab initio*
  algorithms).

Both are passed to every Task in a Recipe. See :ref:`State and
Parameters Objects <state-and-parameters>` for their lifetime and the
``*_name`` keyword convention that protects Tasks from key collisions.

----

How the components fit together
===============================

The Simulation object aggregates the Model object and the Algorithm
object; the Dynamics Driver accepts the Simulation object as input,
creates the State and Parameters objects, runs the dynamics, and
returns the Data object. Tasks and Ingredients populate the Algorithm
object and the Model object respectively.

.. container:: graphviz-center

   .. graphviz::

      digraph flow {
        rankdir=TB;
        bgcolor="transparent";
        node [
          fontsize=12
          fontname="Helvetica, Arial, sans-serif"
          margin="0.3,0.2"
          style=filled
          fillcolor=white
          color="#f38c3c"
        ];

        sim   [label="Simulation Object",  URL="simulation.html"];
        model [label="Model Object",       URL="model.html"];
        algo  [label="Algorithm Object",   URL="algorithm.html"];
        driver[label="Dynamics Driver",    URL="driver.html"];
        data  [label="Data Object",        URL="data.html"];
        ingredients [label="Ingredients",  URL="ingredient.html"];
        tasks [label="Tasks",              URL="task.html"];

        ingredients -> model [color="#f38c3c"];
        tasks -> algo [color="#f38c3c"];
        model -> sim [color="#f38c3c"];
        algo  -> sim [color="#f38c3c"];
        sim   -> driver [color="#f38c3c"];
        driver-> data [color="#f38c3c"];
      }

Each node in the diagram links to its dedicated section.

----

Building blocks
===============

Three named concepts populate the Model object and the Algorithm
object.

Ingredients
-----------

An Ingredient is a function with signature
``f(model, parameters, **kwargs)`` that returns a single physically
meaningful quantity — a Hamiltonian, a gradient, an initialization, a
hop test. Ingredients are attached to the Model object as
``(slot_name, callable)`` tuples in ``model.ingredients``. The list is
consulted back-to-front, so appending ``("h_qc", my_new_h_qc)``
overrides the existing Ingredient in the ``h_qc`` slot. The
comprehensive list of slot names lives in
:ref:`Conventions <conventions>`; the Ingredient mechanism is covered
in :ref:`Ingredients <ingredient>`.

Tasks
-----

A Task is a function with signature
``f(sim, state, parameters, **kwargs)`` returning
``(state, parameters)``. A Task reads named entries from the State and
Parameters objects, calls Ingredients via
:meth:`Model.get <qclab.Model.get>`, and writes named entries back. The
``*_name`` keyword convention lets a single Task be reused under
different State entries by wrapping it in :func:`functools.partial`.
Tasks fall into three categories — initialization Tasks, update Tasks,
and collect Tasks — corresponding to the three Recipes. See
:ref:`Tasks <task>`.

Recipes
-------

A Recipe is a chronological list of Tasks held on the Algorithm
object. Every Algorithm object exposes three Recipes: the
initialization Recipe (run once at ``t = 0``), the update Recipe (run
on every update time step), and the collect Recipe (run on every
collect time step). The full Recipe machinery — including the
deep-copy boundary at :meth:`Algorithm.__init__ <qclab.Algorithm.__init__>`
and the ``recipe = recipe + [task]`` editing idiom — is covered in
:ref:`Recipes <recipe>`.

Adding new physics or a new dynamics method means writing the
necessary Tasks and Ingredients and then modifying the Recipes and
the Ingredient list. See :ref:`Developing Models and Ingredients
<developing-models>` and :ref:`Developing Custom Algorithms
<developing-algorithms>` for the two cases.

----

The dynamics flow
=================

A simulation is run by handing a populated Simulation object to one of
the three Dynamics Drivers in :mod:`qclab.dynamics`:
:func:`~qclab.dynamics.serial_driver`,
:func:`~qclab.dynamics.parallel_driver_multiprocessing`, or
:func:`~qclab.dynamics.parallel_driver_mpi`. The Driver divides
``num_trajs`` into batches of size ``batch_size``, builds a fresh
State object and a fresh Parameters object for each batch, and calls the dynamics
core :func:`qclab.dynamics.run_dynamics` on each batch. Per-batch Data
objects are then merged into a single Data object by
:meth:`Data.add_data <qclab.data.Data.add_data>` using a
trajectory-count-weighted average. See :ref:`Drivers <driver>` for the
three Drivers and :ref:`Parallelization and Reproducibility
<parallelization>` for the cross-cutting topics (batch size, seeds,
MPI specifics).

Inside a single batch, :func:`qclab.dynamics.run_dynamics` iterates
over update time steps. At each update time step:

#. If the time index is ``0``, the initialization Recipe runs once to
   populate the State and Parameters objects with the wavefunction,
   the complex-classical coordinate, and any algorithm-specific
   entries.
#. If the time index is a multiple of ``dt_collect_n``, the collect
   Recipe runs. Its final entries — the contents of
   ``state["output_dict"]`` — are then summed across the batch axis
   and divided by the running normalization factor by
   :meth:`Data.add_output_to_data_dict <qclab.data.Data.add_output_to_data_dict>`,
   producing one trajectory-averaged row in
   ``data.data_dict[<key>]`` per collect time step.
#. The update Recipe runs unconditionally, advancing the wavefunction
   and the complex-classical coordinate by ``dt_update``.

The two granularities — ``dt_update`` for the integrator and
``dt_collect`` for the recorded output — are independently
configurable on ``sim.settings``.

----

Cross-compatibility of Models and Algorithms
============================================

A diabatic Algorithm runs against any Model object defined in a
diabatic basis. The *ab initio* Algorithms
:class:`~qclab.algorithms.MeanFieldAbInitio` and
:class:`~qclab.algorithms.FewestSwitchesSurfaceHoppingAbInitio` pair
only with the :class:`~qclab.models.AbInitio` Model object, and vice
versa; mixing the two families is not supported. See
:ref:`Choosing an Algorithm <choosing-algorithm>` for the
compatibility matrix and a decision tree.

----

Numerical kernels and tunable thresholds
========================================

The :mod:`qclab.functions` module collects the low-level math used by
the built-in Ingredients and Tasks: the complex-coordinate conversions
(``z_to_q``, ``z_to_p``, ``qp_to_z``, ``dqdp_to_dzc``,
``dzdzc_to_dqdp``), batched matrix-vector helpers, RK4 sub-step
kernels, the sparse-gradient inner product, gauge-fixing routines, and
the numerical fewest-switches surface-hopping hop test. Hot loops are
decorated with ``@njit`` from :mod:`qclab.utils`, which falls back to a
no-op when Numba is unavailable. The full reference is in
:ref:`Low-level Functions <functions>`.

The :mod:`qclab.numerical_constants` module holds numerical thresholds
(``SMALL``, ``GAUGE_FIX_THRESHOLD``, ``FINITE_DIFFERENCE_DELTA``) and
unit-conversion factors used by the built-in Model objects. See
:ref:`Numerical Constants <numerical-constants>`.

Optional dependencies are gated by the ``DISABLE_NUMBA``,
``DISABLE_H5PY``, and ``DISABLE_ASE`` flags in :mod:`qclab.utils`, so
QC Lab installs and runs without any of them. Features that depend on
each are degraded accordingly.

----

Module map
==========

The following tree is comprehensive for the top-level layout of
``src/qclab/``.

.. code-block:: text

    src/qclab/
    ├── __init__.py               # Top-level imports and version
    ├── simulation.py             # Simulation class
    ├── model.py                  # Model base class and Ingredient lookup
    ├── algorithm.py              # Algorithm base class and Recipe executor
    ├── constants.py              # Constants attribute-bag with change hook
    ├── data.py                   # Data: collection, merging, HDF5 / npz I/O
    ├── utils.py                  # JIT shims, in-memory logging, optional-dep flags
    ├── numerical_constants.py    # SMALL, finite-difference delta, unit conversions
    ├── ingredients.py            # Built-in Ingredients
    ├── functions.py              # Low-level numerics, JIT kernels, gauge fixing
    ├── algorithms/               # MeanField, FSSH, and the *ab initio* variants
    ├── dynamics/                 # run_dynamics core + serial / MP / MPI Drivers
    ├── models/                   # SpinBoson, Holstein, FMOComplex, Tully I / II / III, AbInitio
    ├── tasks/                    # Initialization, update, and collect Tasks
    └── interfaces/               # Q-Chem *ab initio* Interface
