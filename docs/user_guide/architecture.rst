.. _architecture:

==========================
Architecture Overview
==========================

QC Lab is built around five objects and a single integration core. This
page walks through that architecture from the top down, so that readers
can locate any new feature inside the codebase without having to discover
the layout by trial and error.

For a one-line summary: a ``Simulation`` holds a ``Model`` and an
``Algorithm`` and is handed to a driver. The driver iterates a small
loop in :func:`qclab.dynamics.run_dynamics` that executes the
``Algorithm``'s three task lists. Tasks read named entries from a
``state`` dict, call physics functions ("ingredients") on the ``Model``,
and write named entries back. Trajectory-averaged outputs end up in a
``Data`` object.

The five objects
================

.. list-table::
   :header-rows: 1
   :widths: 18 50 32

   * - Object
     - Role
     - Defined in
   * - ``Simulation``
     - Top-level container; holds settings, the Model, the Algorithm,
       the initial-state dict, and the per-run time index ``t_ind``.
     - ``simulation.py``
   * - ``Model``
     - Physical system. Holds ``constants`` and a list of ingredients
       (callables) that define its Hamiltonians and initialization.
     - ``model.py``
   * - ``Algorithm``
     - Numerical recipe. Holds three task lists —
       ``initialization_recipe``, ``update_recipe``, and
       ``collect_recipe`` — plus algorithm-specific settings.
     - ``algorithm.py``
   * - ``Constants``
     - Attribute-bag with a ``__setattr__`` hook that re-runs a
       registered initializer whenever a constant changes after init.
       Used by both ``Model`` and ``Algorithm``.
     - ``constants.py``
   * - ``Data``
     - Trajectory-averaged output container. Supports HDF5 / ``.npz``
       I/O, log capture, and incremental merging via
       :meth:`~qclab.data.Data.add_data`.
     - ``data.py``

A simulation is run by handing the ``Simulation`` to one of the drivers
in :mod:`qclab.dynamics`: :func:`~qclab.dynamics.serial_driver`,
:func:`~qclab.dynamics.parallel_driver_multiprocessing`, or
:func:`~qclab.dynamics.parallel_driver_mpi`. Each driver builds per-batch
``state``/``parameters`` dicts, runs the dynamics core, and merges the
resulting ``Data`` objects.

The relationships between the objects are summarised graphically below
(click any node to jump to the corresponding page):

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

        sim   [label="Simulation Object", URL="simulation.html"];
        model [label="Model Object",      URL="model.html"];
        algo  [label="Algorithm Object",  URL="algorithm.html"];
        driver[label="Dynamics Driver",   URL="driver.html"];
        data  [label="Data Object",       URL="data.html"];
        ingredients [label="Ingredients", URL="ingredient.html"];
        tasks [label="Tasks", URL="task.html"];

        ingredients -> model [color="#f38c3c"];
        tasks -> algo [color="#f38c3c"];
        model -> sim [color="#f38c3c"];
        algo  -> sim [color="#f38c3c"];
        sim   -> driver [color="#f38c3c"];
        driver-> data [color="#f38c3c"];
      }

----

The ingredient / task / recipe pattern
======================================

Behind the five-object skeleton, every simulation is assembled from three
layers. Keeping these layers separate is what allows new physics to be
swapped in without touching the integrator:

**Ingredients** are physics functions ``f(model, parameters, **kwargs)``
that compute a single quantity (a Hamiltonian, a gradient, an initial
condition, a hop result). They live on the Model as ``(slot_name,
callable)`` tuples. The list is consulted back-to-front, so appending
``("h_qc", my_new_h_qc)`` overrides the existing ``h_qc`` ingredient
without removing it. The full list of standard slot names lives on
:ref:`Conventions <conventions>`.

**Tasks** are algorithm steps ``task(sim, state, parameters, **opts) ->
(state, parameters)``. They read named entries from ``state`` /
``parameters``, call ingredients via :meth:`Model.get
<qclab.Model.get>`, and write named entries back. Every state key a
task touches is exposed as a ``*_name`` keyword argument so the same
task can be reused across recipes via ``functools.partial``. See
:ref:`State and Parameters <state-and-parameters>` for the rebinding
pattern.

**Recipes** are plain Python lists of tasks on the Algorithm class. The
:meth:`Algorithm.execute_recipe <qclab.Algorithm.execute_recipe>` method
just iterates the list and threads ``(state, parameters)`` through each
call. Algorithms inherit from :class:`qclab.Algorithm`, which exposes
three recipe attributes: ``initialization_recipe`` (run once),
``update_recipe`` (run every step), and ``collect_recipe`` (run only on
collect steps).

This gives a clean rule of thumb:

- A *new algorithm* is a new ordering of existing tasks.
- A *new model* is a new ingredient list.
- A *new physics ingredient* is the only thing that should require
  writing genuinely new code at the lowest level.

----

The dynamics core
=================

The integration loop is a single function,
:func:`qclab.dynamics.run_dynamics`. It is intentionally small. Each
step it does the following:

1. On ``t_ind == 0``, run ``initialization_recipe``.
2. At every collect step (``t_ind % dt_collect_n == 0``), run
   ``collect_recipe`` and call
   :meth:`Data.add_output_to_data_dict <qclab.data.Data.add_output_to_data_dict>`
   to merge ``state["output_dict"]`` into the Data object.
3. On every step, run ``update_recipe``.

Drivers wrap this core. They divide ``num_trajs`` across batches of size
``batch_size``, build a fresh ``state``/``parameters`` dict per batch,
call ``run_dynamics``, and then merge per-batch Data objects via
:meth:`Data.add_data <qclab.data.Data.add_data>` (which uses a
trajectory-count-weighted average). See :ref:`Drivers <driver>` for the
serial / multiprocessing / MPI variants.

----

Numerical kernels
=================

:mod:`qclab.functions` collects the low-level math used by ingredients
and tasks: complex-coordinate conversions (``z_to_q``, ``qp_to_z``,
``dqdp_to_dzc``, ``dzdzc_to_dqdp``), batched matrix-vector helpers,
RK4 sub-step kernels, the sparse-gradient inner product
``calc_sparse_inner_product``, gauge fixing, and ``numerical_fssh_hop``.
Hot loops are decorated with ``@njit`` (a Numba shim from
:mod:`qclab.utils` that becomes a no-op when Numba is unavailable). See
:ref:`Functions <functions>` for the full reference and
:ref:`Numerical Constants <numerical-constants>` for the unit conversions
and tunable thresholds.

Optional dependencies are handled via the ``DISABLE_NUMBA``,
``DISABLE_H5PY``, and ``DISABLE_ASE`` flags in :mod:`qclab.utils`, so
QC Lab installs and runs without any of them — features that depend on
each are gracefully degraded.

----

Module map
==========

.. code-block:: text

    src/qclab/
    ├── __init__.py               # Top-level imports and version
    ├── simulation.py             # Simulation class
    ├── model.py                  # Model base class and ingredient lookup
    ├── algorithm.py              # Algorithm base class and recipe executor
    ├── constants.py              # Constants attribute-bag with change hook
    ├── data.py                   # Data: collection, merging, HDF5 / npz I/O
    ├── utils.py                  # JIT shims, in-memory logging, optional-dep flags
    ├── numerical_constants.py    # SMALL, finite-difference delta, unit conversions
    ├── ingredients.py            # Reusable model ingredients
    ├── functions.py              # Low-level numerics, JIT kernels, gauge fixing
    ├── algorithms/               # MeanField, FSSH (and ab initio variants)
    ├── dynamics/                 # run_dynamics core + serial / MP / MPI drivers
    ├── models/                   # Spin-boson, Holstein, FMO, Tully I / II / III, AbInitio
    ├── tasks/                    # Initialization, update, and collect tasks
    └── interfaces/               # Q-Chem ab initio interface
