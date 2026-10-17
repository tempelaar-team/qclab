.. _architecture:

==========================
Architecture Overview
==========================

A QC Lab simulation consists primarily of a Model object and an
Algorithm object, held inside a Simulation object and executed by a
Dynamics Driver. The Driver returns a Data object containing the
trajectory-averaged results. This section outlines the components,
shows how they fit together, and traces the path a single batch of
trajectories takes from input to output.

----

Components
==========

A QC Lab simulation is described by a total of five objects with
well-defined responsibilities.

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
     - Container for named constants and settings.
     - ``constants.py``
   * - Data object
     - Trajectory-averaged output container. Stores results in
       ``data.data_dict``, captures the in-memory log, and can be
       saved to and loaded from HDF5 or ``.npz``.
     - ``data.py``

Two further dictionaries — the **State object** and the
**Parameters object** — are utilized by the Dynamics Driver. See
:ref:`State and Parameters <state-and-parameters>` for
details.

----

How the components fit together
===============================

The diagram below summarizes how the components fit together; each
node links to its dedicated section.

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

----

Building blocks
===============

Adding new physics or a new dynamics method means writing the
necessary Tasks and Ingredients and then modifying the Recipes and
the Ingredient list.

The Model object and the Algorithm object are populated by
Ingredients and Tasks. The Algorithm executes Tasks in chronological
lists called Recipes.

Ingredients
-----------

An Ingredient is a function with signature
``f(model, parameters, **kwargs)`` that returns a physically
meaningful quantity (a Hamiltonian, a gradient, an initialization, or
a hop test). Some Ingredients — the sparse gradients in particular —
return a tuple such as ``(inds, mels, shape)``. Ingredients are
attached to the Model object as ``(slot_name, callable)`` tuples in
``model.ingredients``. The list is consulted back-to-front, so
appending ``("h_qc", my_new_h_qc)`` overrides the existing Ingredient
in the ``h_qc`` slot. See :ref:`Ingredients <ingredient>`.

Tasks and Recipes
-----------------

A Task is a function with signature
``f(sim, state, parameters, **kwargs)`` returning
``(state, parameters)``. A Task reads named entries from the State
and Parameters objects, calls Ingredients via
:meth:`Model.get <qclab.Model.get>`, and writes named entries back.
The ``*_name`` keyword convention lets a single Task be reused under
different State entries by wrapping it in :func:`functools.partial`.
Tasks fall into three categories — initialization Tasks, update
Tasks, and collect Tasks — corresponding to the three Recipes.

A Recipe is a chronological list of Tasks held on the Algorithm
object. Every Algorithm object exposes three Recipes: the
initialization Recipe runs once before the dynamics start, the update
Recipe runs on every update time step, and the collect Recipe runs on
every collect time step. See :ref:`Tasks <task>`.

----

The dynamics flow
=================

A simulation is run by handing a populated Simulation object to one
of the three Dynamics Drivers in :mod:`qclab.dynamics`:
:func:`~qclab.dynamics.serial_driver`,
:func:`~qclab.dynamics.parallel_driver_multiprocessing`, or
:func:`~qclab.dynamics.parallel_driver_mpi`. The Driver divides
``num_trajs`` into batches of size ``batch_size``, builds a fresh
State object and a fresh Parameters object for each batch, and calls
the dynamics core :func:`qclab.dynamics.run_dynamics` on each batch.
Per-batch Data objects are then merged into a single Data object by
:meth:`Data.add_data <qclab.data.Data.add_data>` using a
trajectory-count-weighted average. See :ref:`Drivers <driver>`.

Inside a single batch, :func:`qclab.dynamics.run_dynamics` iterates
over update time steps:

#. Before the dynamics start, the initialization Recipe runs once to
   populate the State and Parameters objects with the wavefunction,
   the complex-valued classical coordinate, and any
   algorithm-specific entries.
#. On every collect time step, the collect Recipe runs. Its final
   entries — the contents of ``state["output_dict"]`` — are then
   summed across the batch axis and divided by the running
   normalization factor by
   :meth:`Data.add_output_to_data_dict <qclab.data.Data.add_output_to_data_dict>`,
   producing one trajectory-averaged row in
   ``data.data_dict[<key>]`` per collect time step.
#. On every update time step, the update Recipe runs, advancing the
   wavefunction and the complex-valued classical coordinate by
   ``dt_update``.

The two granularities — ``dt_update`` for the integrator and
``dt_collect`` for the recorded output — are independently
configurable on ``sim.settings``.

----

Cross-compatibility of Models and Algorithms
============================================

A diabatic Algorithm runs against any Model object defined in a
diabatic basis. The *ab initio* Algorithms pair only with the
*ab initio* Model objects, and vice versa; mixing the two families is
not supported.

----

Numerical kernels and tunable thresholds
========================================

The :mod:`qclab.functions` module collects the low-level math used by
the Ingredients and Tasks shipped with QC Lab: the conversions
between the complex-valued classical coordinate and the real-valued
``(q, p)`` pair (``z_to_q``, ``z_to_p``, ``qp_to_z``,
``dqdp_to_dzc``, ``dzdzc_to_dqdp``), batched matrix-vector helpers,
RK4 sub-step kernels, the sparse-gradient inner product, gauge-fixing
routines, and the numerical fewest-switches surface-hopping hop test.
Hot loops are decorated with ``@njit`` from :mod:`qclab.utils`, which
falls back to a no-op when Numba is unavailable.

The :mod:`qclab.numerical_constants` module holds numerical
thresholds (``SMALL``, ``GAUGE_FIX_THRESHOLD``,
``FINITE_DIFFERENCE_DELTA``) and unit-conversion factors. See
:ref:`Numerical Constants <numerical-constants>`.

Optional dependencies are gated by the ``DISABLE_NUMBA``,
``DISABLE_H5PY``, and ``DISABLE_ASE`` flags in :mod:`qclab.utils`, so
QC Lab installs and runs without any of them. Features that depend
on each are degraded accordingly.

----

Module map
==========

The following tree is comprehensive for the top-level layout of
``src/qclab/``.

.. code-block:: text

    src/qclab/
    ├── __init__.py               # Top-level imports and version
    ├── simulation.py             # Simulation class
    ├── model.py                  # Model class
    ├── algorithm.py              # Algorithm class
    ├── constants.py              # Constants class
    ├── data.py                   # Data class
    ├── utils.py                  # Internal utilities
    ├── numerical_constants.py    # Numerical constants
    ├── ingredients.py            # Ingredients shipped with QC Lab
    ├── functions.py              # Low-level math helpers
    ├── algorithms/               # Algorithm subclasses
    ├── dynamics/                 # Dynamics core and Drivers
    ├── models/                   # Model subclasses
    ├── tasks/                    # Initialization, update, and collect Tasks
    └── interfaces/               # *ab initio* Interfaces
