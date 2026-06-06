.. _parallelization:

==========================================
Parallelization and Reproducibility
==========================================

QC Lab ships with three Dynamics Drivers — a serial Driver and two
parallel Drivers — that differ in how they distribute batches of
trajectories across cores or nodes. This section is the companion to
:ref:`Drivers <driver>` and covers the cross-cutting topics that span
all three Drivers: choosing the right Driver for a workload, how
``batch_size``, ``num_trajs``, and ``num_tasks`` interact, how seeds
flow through batches, and how to reproduce a single trajectory.

----

Choosing a Dynamics Driver
==========================

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Driver
     - When to use
     - Limitations
   * - ``serial_driver``
     - Small simulations, debugging, profiling, or when an external
       library (e.g., Q-Chem) manages its own parallelism.
     - Single Python process; does not actively limit CPU usage but
       does not distribute batches across cores.
   * - ``parallel_driver_multiprocessing``
     - Single-machine production runs. Uses Python's
       :mod:`multiprocessing` module to run batches concurrently
       across CPU cores.
     - Pickle requirements on the Simulation object; not portable
       across machines.
   * - ``parallel_driver_mpi``
     - Multi-node runs on HPC clusters. Uses ``mpi4py`` and integrates
       with ``mpirun`` / ``mpiexec`` and SLURM.
     - Requires ``mpi4py`` to be installed and an MPI execution
       environment to launch the script.

A reasonable progression during development is:

#. Build the Simulation with ``serial_driver`` and a small
   ``num_trajs``.
#. Once correctness is established, switch to
   ``parallel_driver_multiprocessing`` for full-scale runs on a
   single machine.
#. Move to ``parallel_driver_mpi`` only when a single machine is no
   longer enough.

----

How batches, trajectories, and tasks fit together
=================================================

Three settings determine how a simulation is partitioned for
execution:

``sim.settings.num_trajs``
    Total number of independent trajectories to run, summed across
    all batches and all tasks.

``sim.settings.batch_size``
    Number of trajectories carried inside a single batch. A batch is
    the unit that the dynamics core processes vectorized along the
    leading axis of every State entry. Each call to ``run_dynamics``
    consumes one batch.

``num_tasks`` (parallel Drivers only)
    Number of parallel workers — multiprocessing processes for
    ``parallel_driver_multiprocessing``, MPI ranks for
    ``parallel_driver_mpi``. Defaults to
    :func:`multiprocessing.cpu_count` and ``MPI.COMM_WORLD.Get_size``
    respectively.

These three settings combine in the following way. The serial Driver
splits ``num_trajs`` into ``ceil(num_trajs / batch_size)`` batches
and runs them in sequence. Each parallel Driver does the same split
and then distributes the batches across ``num_tasks`` workers; a
single worker can hold up to ``batch_size`` trajectories in memory at
once.

The practical rules of thumb are:

- Set ``batch_size`` to fit comfortably in one worker's memory. The
  State object grows linearly in ``batch_size`` and ``num_quantum_states``;
  for FSSH on a 7-site FMO model with ``batch_size = 100`` the State
  fits in ~ tens of MB. Vectorization within a batch is where the
  per-core performance comes from, so larger batches are generally
  preferable to many small batches.
- Set ``num_trajs`` to be a multiple of ``num_tasks * batch_size``.
  When the last partial batch is smaller than ``batch_size`` the
  parallel Driver still distributes one batch per worker, so a
  partially filled batch ties up a whole worker for fewer
  trajectories than the rest. Choosing a multiple avoids this
  inefficiency.
- Set ``num_tasks`` to the number of physical cores you intend to
  use. Oversubscribing — ``num_tasks`` greater than the actual core
  count — usually slows the simulation because NumPy and any
  underlying BLAS library are already using threads inside each
  batch (see :ref:`Thread interference <parallelization-threads>`).

----

Seeds and reproducibility
=========================

Every trajectory has an integer seed. The seeds are stored in
``state["seed"]`` and used by every Task that draws random numbers,
so the same seed array always produces the same trajectory data,
batch by batch, regardless of the Driver chosen.

Default seed policy
-------------------

When ``seeds`` is not passed to the Driver, the Driver constructs the
seed array deterministically from the current Data object's seed
record:

.. code-block:: python

    if seeds is None:
        if len(data.data_dict["seed"]) > 0:
            offset = np.max(data.data_dict["seed"]) + 1
        else:
            offset = 0
        seeds = offset + np.arange(sim.settings.num_trajs, dtype=int)

If a fresh Data object is passed (or no ``data`` argument is given),
the seeds are ``0, 1, …, num_trajs - 1``. If the Data object already
contains trajectories with seeds ``0, …, N - 1``, the new run uses
``N, N + 1, …, N + num_trajs - 1``. This makes ``add_data`` calls
non-overlapping in seed space and lets a user grow a Data object
incrementally without seed collisions.

Reproducing a single trajectory
-------------------------------

Passing ``seeds`` explicitly to the Driver overrides the default
policy and lets a single trajectory be re-run by itself:

.. code-block:: python

    # Reproduce trajectory 17 from an earlier run.
    sim.settings.batch_size = 1
    sim.settings.num_trajs = 1
    data_single = serial_driver(sim, seeds=np.array([17]))

When ``seeds`` is provided, the Driver sets ``sim.settings.num_trajs``
to ``len(seeds)``; the script does not need to set ``num_trajs``
separately.

For deterministic reproducibility, two further conditions must hold:

- Every random draw inside a Task or Ingredient must derive from
  ``state["seed"]`` (the built-in Tasks do this).
- The Algorithm's deterministic-flag settings must match between the
  original run and the reproduction. For FSSH, this is
  ``fssh_deterministic``.

The ``seed`` array is also written into ``data.data_dict["seed"]``
after a run, so the seeds used in any saved Data object are
recoverable.

----

The MPI Driver
==============

``parallel_driver_mpi`` requires ``mpi4py`` and an MPI execution
environment. A script using the MPI Driver is launched as

.. code-block:: bash

    mpirun -n 4 python my_simulation_script.py

with ``-n 4`` requesting four MPI ranks. The Driver distributes the
batches across ranks via ``np.linspace(0, num_batches, size + 1)``
chunking, runs the dynamics on each rank, and then gathers the
results onto rank 0 via point-to-point sends. Log output from every
rank is concatenated onto ``data.log`` on rank 0.

A complete SLURM submission script and a worked example live in
``examples/mpi_examples/``; see :ref:`Drivers <driver>` for the
embedded listing.

Two MPI-specific caveats:

- ``data.add_data`` is called only on rank 0. The other ranks send
  their local Data objects via ``comm.send`` and exit; the Data
  object returned by the Driver on non-zero ranks contains only the
  rank-local batches.
- The MPI Driver imports ``mpi4py`` lazily inside the function body.
  A script that defines an MPI-targeted simulation but is invoked
  without MPI will not fail until the Driver is called.

----

.. _parallelization-threads:

Common pitfalls
===============

Oversubscribed cores
--------------------

NumPy and any underlying BLAS library (OpenBLAS, MKL) often use
threads inside a single Python process. When ``num_tasks`` equals the
core count and each worker is also running BLAS at full thread
count, the cores are oversubscribed. The usual remedy is to set the
thread count to one before launching the Driver:

.. code-block:: bash

    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    python my_simulation_script.py

Non-deterministic random in custom Ingredients
----------------------------------------------

A custom Ingredient that uses :func:`numpy.random.default_rng` with
no seed argument will draw from a global generator and break
trajectory-level reproducibility. The pattern that preserves
determinism is to seed the generator from the trajectory seed:

.. code-block:: python

    def init_classical_custom(model, parameters, seed=None, **kwargs):
        rng = np.random.default_rng(seed)
        return rng.normal(size=(len(seed), model.constants.num_classical_coordinates)) + 0j

The ``seed`` keyword argument is passed by the calling Task and
contains the per-trajectory seed array; threading it through
``default_rng`` keeps every trajectory independent and reproducible.

Pickle failures in multiprocessing
----------------------------------

The multiprocessing Driver deep-copies the Simulation object and
sends each copy to a worker process via pickle. Any callable attached
to the Model object's Ingredients list or the Algorithm object's
Recipes must be picklable; a lambda or a closure defined inside a
function is the most common offender. Define such callables at module
scope, or use :func:`functools.partial` of a module-level function.

Mixed Driver and dataset
------------------------

Switching Drivers between calls is supported, but the Data object's
``norm_factor`` is cumulative: each successive ``add_data`` call
re-weights the running mean by the running and incoming
``norm_factor``. Mixing batches from very different ``batch_size``
values produces a correct average but a less stable estimate than a
homogeneous set of batches; matching ``batch_size`` between calls is
the simpler choice when a saved Data object will be extended.
