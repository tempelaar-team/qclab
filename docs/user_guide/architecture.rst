.. _architecture:

==========================
Architecture Overview
==========================

A simulation in QC Lab is built from five objects and a dynamics core that
integrates them in time. This section describes those five objects, the
way they are populated with physics and numerical operations, and how the
dynamics core consumes them.

At a high level, the Simulation object holds a Model object and an
Algorithm object and is passed to a dynamics driver. The driver loops
over time, executing the three recipes defined on the Algorithm object.
The recipes are lists of tasks that read named entries from the State
object, call ingredients on the Model object, and write named entries
back. Trajectory-averaged outputs end up in a Data object.

The five objects
================

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Object
     - Role
     - Defined in
   * - Simulation object
     - Top-level container. Holds settings, the Model object, the
       Algorithm object, the initial-state dictionary, and the
       per-run time index ``t_ind``.
     - ``simulation.py``
   * - Model object
     - Physical system. Holds a Constants object and a list of
       ingredients that define its Hamiltonians and initialization.
     - ``model.py``
   * - Algorithm object
     - Numerical recipe. Holds three task lists —
       ``initialization_recipe``, ``update_recipe``, and
       ``collect_recipe`` — together with algorithm-specific settings.
     - ``algorithm.py``
   * - Constants object
     - Attribute-bag for constants and settings. Re-runs a registered
       initializer whenever a constant changes after initialization.
       Used by both the Model object and the Algorithm object.
     - ``constants.py``
   * - Data object
     - Trajectory-averaged output container. Supports HDF5 and
       ``.npz`` I/O, captures the in-memory simulation log, and merges
       results from multiple batches via
       :meth:`~qclab.data.Data.add_data`.
     - ``data.py``

A simulation is run by handing the Simulation object to one of the
dynamics drivers in :mod:`qclab.dynamics`:
:func:`~qclab.dynamics.serial_driver`,
:func:`~qclab.dynamics.parallel_driver_multiprocessing`, or
:func:`~qclab.dynamics.parallel_driver_mpi`. Each driver builds per-batch
State and Parameters dictionaries, runs the dynamics core, and merges
the resulting Data objects.

The relationships between the objects are summarised graphically below.
Click any node to jump to the corresponding section.

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

Ingredients, tasks, and recipes
===============================

The physics of a simulation and the algorithm that propagates it are
defined through three related concepts.

Ingredients are functions of the form ``f(model, parameters, **kwargs)``
that compute a single physical quantity — a Hamiltonian, a gradient, an
initial condition, a hop result. They are attached to the Model object
as a list of ``(slot_name, callable)`` tuples in
``model.ingredients``. The list is consulted back-to-front, so
appending ``("h_qc", my_new_h_qc)`` overrides the existing ``h_qc``
ingredient without removing it. The standard slot names are listed in
the :ref:`Conventions <conventions>` section. Ingredients are described
in detail in the :ref:`Ingredients <ingredient>` section.

Tasks are functions of the form
``task(sim, state, parameters, **opts) -> (state, parameters)``. A task
reads named entries from the State and Parameters objects, calls
ingredients via :meth:`Model.get <qclab.Model.get>`, and writes named
entries back. Every State-object key that a task touches is exposed as a
``*_name`` keyword argument with a sensible default, so the same task
can be rebound to different keys with ``functools.partial``. The
:ref:`State and Parameters <state-and-parameters>` section describes
the rebinding pattern. Tasks themselves are described in the
:ref:`Tasks <task>` section.

Recipes are plain Python lists of tasks on the Algorithm class. The
:meth:`Algorithm.execute_recipe <qclab.Algorithm.execute_recipe>` method
iterates the list and threads ``(state, parameters)`` through each call.
The Algorithm object exposes three recipe attributes:
``initialization_recipe`` is executed once at the start of the
simulation, ``update_recipe`` is executed every update time step, and
``collect_recipe`` is executed every collect time step. Algorithms are
described in detail in the :ref:`Algorithms <algorithm>` section.

QC Lab is designed so that new physics and new algorithms can often be
introduced by adding new ingredients to an existing Model object or new
tasks to an existing Algorithm object. Either case may, however,
require writing new ingredients or new tasks as well — for example, a
new algorithm may need a bespoke update task to evaluate a quantity
that the existing tasks do not provide. The :ref:`Developing Models
<developing-models>` section walks through the choices involved when
adding new physics.

----

The dynamics core
=================

The integration loop is implemented by
:func:`qclab.dynamics.run_dynamics`. At each update time step it does
the following.

#. When ``t_ind == 0`` it runs the ``initialization_recipe`` once.
#. On a collect time step (``t_ind % dt_collect_n == 0``), it runs the
   ``collect_recipe`` and then calls
   :meth:`Data.add_output_to_data_dict <qclab.data.Data.add_output_to_data_dict>`
   to merge ``state["output_dict"]`` into the Data object.
#. On every update time step, it runs the ``update_recipe``.

The dynamics drivers wrap this core. A driver divides ``num_trajs``
across batches of size ``batch_size``, builds a fresh State and
Parameters dictionary for each batch, calls ``run_dynamics``, and then
merges the per-batch Data objects via
:meth:`Data.add_data <qclab.data.Data.add_data>`. The merge uses a
trajectory-count-weighted average. See :ref:`Drivers <driver>` for the
serial, multiprocessing, and MPI variants.

----

Numerical kernels
=================

The :mod:`qclab.functions` module collects the low-level math used by
ingredients and tasks. This includes the complex-coordinate conversions
(``z_to_q``, ``qp_to_z``, ``dqdp_to_dzc``, ``dzdzc_to_dqdp``), batched
matrix-vector helpers, RK4 sub-step kernels, the sparse-gradient inner
product ``calc_sparse_inner_product``, gauge fixing, and
``numerical_fssh_hop``. Hot loops are decorated with ``@njit`` from
:mod:`qclab.utils`, which falls back to a no-op when Numba is
unavailable. See :ref:`Functions <functions>` for the full reference and
:ref:`Numerical Constants <numerical-constants>` for the unit
conversions and tunable thresholds.

Optional dependencies are handled by the ``DISABLE_NUMBA``,
``DISABLE_H5PY``, and ``DISABLE_ASE`` flags in :mod:`qclab.utils`, so
QC Lab installs and runs without any of them. Features that depend on
each are degraded accordingly.

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
