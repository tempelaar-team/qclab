.. _state-and-parameters:

==================================
The State and Parameters Dicts
==================================

Every QC Lab task has the same signature::

    def my_task(sim, state, parameters, **kwargs):
        ...
        return state, parameters

The ``state`` and ``parameters`` arguments are plain Python dictionaries
that are threaded through every task in a recipe by
:meth:`~qclab.Algorithm.execute_recipe`. They are the bus that ties the
pipeline of tasks together.

.. note::

    A complete inventory of standard keys (``wf_db``, ``z``, ``h_q_tot``, etc.)
    lives on :ref:`Conventions <conventions>`. This page explains how the
    dicts are used; the conventions page tells you what to call things.

----

Two dicts, two lifetimes
========================

Both dictionaries are created at the start of every batch and discarded at
the end. They are *not* shared across batches.

``state``
    Holds quantities that are intrinsic to the current batch of trajectories
    and that change every step: the wavefunction, the classical coordinate,
    eigenvectors, density matrices, and so on. Tasks read named entries from
    ``state``, perform their computation, and write named entries back. All
    state entries have a leading batch axis of length ``sim.settings.batch_size``.

``parameters``
    Holds quantities that are computed once and then used many times within
    a batch, or quantities that come from the *outside* of the trajectory
    pipeline (for example, the cached output of an electronic-structure
    calculation in the ab initio algorithms).

The dynamics core in :func:`qclab.dynamics.run_dynamics` does not look at
either dict directly; it just hands them to whichever recipe is being
executed:

.. code-block:: python

    # Roughly what run_dynamics does each step:
    state, parameters = sim.algorithm.execute_recipe(
        sim, state, parameters, sim.algorithm.update_recipe
    )

Because the dicts are shared by every task in the recipe, name collisions
matter. Tasks that read or write a quantity should use the standard key
listed in :ref:`Conventions <conventions>` whenever the quantity already
has a name there.

----

The ``output_dict`` bridge to Data
==================================

Only one entry of ``state`` ever leaves the trajectory pipeline:
``state["output_dict"]``. Whatever lives there at the end of a collect
step is averaged across the batch by
:meth:`~qclab.data.Data.add_output_to_data_dict` and stored in the Data
object returned by the driver.

This means that, in QC Lab, recording a new quantity is a two-step
process:

1. **Compute** the quantity in an *update task* and store the result on
   ``state``.
2. **Copy** the quantity into ``state["output_dict"]`` from a separate
   *collect task*.

Doing both steps in a single task is technically possible but discouraged.
Update tasks run on every step; collect tasks run only on collect steps.
Splitting the work is also what allows expensive observables (e.g.,
projections onto adiabatic surfaces) to be computed at the natural
frequency.

A minimal example pair (mean classical position) looks like this:

.. code-block:: python

    import numpy as np
    from qclab.functions import z_to_q

    def update_mean_position(sim, state, parameters,
                             *, z_name="z",
                             mean_position_name="mean_position"):
        """Compute the mean real-space position from the complex coordinate."""
        z = state[z_name]
        m = sim.model.constants.classical_coordinate_mass[np.newaxis, :]
        h = sim.model.constants.classical_coordinate_weight[np.newaxis, :]
        q = z_to_q(z, m, h)
        state[mean_position_name] = q.mean(axis=1)
        return state, parameters

    def collect_mean_position(sim, state, parameters,
                              *, mean_position_name="mean_position",
                              mean_position_output_name="mean_position"):
        """Copy the mean position into output_dict for recording."""
        state["output_dict"][mean_position_output_name] = state[mean_position_name]
        return state, parameters

To wire these into a simulation, append them *after* the algorithm has
been instantiated:

.. code-block:: python

    sim.algorithm.initialization_recipe = (
        sim.algorithm.initialization_recipe + [update_mean_position]
    )
    sim.algorithm.update_recipe = (
        sim.algorithm.update_recipe + [update_mean_position]
    )
    sim.algorithm.collect_recipe = (
        sim.algorithm.collect_recipe + [collect_mean_position]
    )

The update task is included in the initialization recipe so that the
quantity exists at ``t = 0`` before the first collect step runs. After
the simulation, the new key shows up in
``data.data_dict["mean_position"]``.

.. warning::

    Do not modify ``algorithm.collect_recipe`` (or the other recipes) by
    in-place ``.append`` *before* the algorithm is instantiated.
    ``Algorithm.__init__`` deep-copies the class attribute, so an
    in-place change on the class can be lost. Always use the
    ``recipe = recipe + [task]`` pattern *after* instantiation.

----

The ``*_name`` keyword convention
=================================

Tasks must not hard-code state-dict keys. Every key a task reads or writes
is exposed as a ``*_name`` keyword argument with a sensible default. This
is what lets recipes rebind the same task to different keys via
``functools.partial``:

.. code-block:: python

    from functools import partial
    from qclab import tasks

    # The same task, rebound to read "z_1" instead of "z":
    rk4_step_two = partial(
        tasks.update_z_rk4_k123,
        z_name="z",
        z_k_name="z_2",
        k_name="z_rk4_k2",
    )

This is exactly how the built-in mean-field and FSSH algorithms reuse a
single set of RK4 building blocks across the four sub-steps; see
``qclab/algorithms/mean_field.py`` for the full pattern.

When you write a new task, follow the same rule:

- Every state key the task reads is a ``*_name`` kwarg.
- Every state key the task writes is a ``*_name`` kwarg.
- Defaults match the standard names from :ref:`Conventions <conventions>`.

----

Task docstring sections
=======================

The Sphinx configuration adds a handful of custom Napoleon sections that
QC Lab tasks use to document their inputs and outputs. When you write a
new task, follow the same template — Sphinx renders these sections as
parameter-style lists.

``Optional Keyword Arguments``
    The ``*_name`` keyword arguments accepted by the task and their
    defaults.

``Reads``
    Entries of ``state`` (or ``parameters``) that the task expects to
    already exist. Include the shape and dtype.

``Writes``
    Entries of ``state`` (or ``parameters``) that the task creates or
    overwrites. Include the shape and dtype.

``Constants and Settings``
    Entries of ``sim.model.constants`` or ``sim.settings`` that the task
    consults.

``Ingredients``
    Standard slot names that the task calls via ``sim.model.get(...)``.

``Shapes and dtypes``
    Free-form notes about shape conventions when they are not obvious from
    Reads/Writes.

A typical collect-task docstring looks like:

.. code-block:: rst

    Optional Keyword Arguments
    --------------------------
    t_name:
        Name of the time variable in the State object.
    t_output_name:
        Name of the time variable in the output dictionary.

    Reads
    -----
    state[t_name]: ndarray of shape (B,), dtype=float64
        Time in each trajectory.

    Writes
    ------
    state["output_dict"][t_output_name]: ndarray of shape (B,), dtype=float64
        Time in each trajectory.

    Notes
    -----
    * B = sim.settings.batch_size
